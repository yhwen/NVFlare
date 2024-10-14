# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from enum import Enum

from nvflare.app_opt.job_launcher.job_launcher_spec import JobLauncherSpec

from kubernetes import config
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from kubernetes.client.rest import ApiException


class JobState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    TERMINATED = "terminated"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"


POD_STATE_MAPPING = {
    "Pending": JobState.STARTING,
    "Running": JobState.RUNNING,
    "Succeeded": JobState.SUCCEEDED,
    "Failed": JobState.TERMINATED,
    "Unknown": JobState.UNKNOWN
    }


class K8sJobHandle:
    def __init__(self, job_id: str, api_instance: core_v1_api, job_config: dict, namespace='default'):
        super().__init__()
        self.job_id = job_id

        self.api_instance = api_instance
        self.namespace = namespace
        self.pod_manifest = {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': None  # set by job_config['name']
            },
            'spec': {
                'containers': None,  # link to container_list
                'volumes': None  # link to volume_list
            }
        }
        self.volume_list = [
            {
                'name': None,
                'hostPath': {
                    'path': None,
                    'type': 'Directory'
                }
            }
        ]
        self.container_list = [
            {
                'image': None,
                'name': None,
                'command': ['/usr/local/bin/python'],
                'args': None,  # args_list + args_dict + args_sets
                'volumeMounts': None,  # volume_mount_list
                'imagePullPolicy': 'Always'
            }
        ]
        self.container_args_python_args_list = [
            '-u', '-m', 'nvflare.private.fed.app.client.worker_process'
        ]
        self.container_args_module_args_dict = {
            '-m': None,
            '-w': None,
            '-t': None,
            '-d': None,
            '-n': None,
            '-c': None,
            '-p': None,
            '-g': None,
            '-scheme': None,
            '-s': None,
        }
        self.container_volume_mount_list = [
            {
                'name': None,
                'mountPath': None,
            }
        ]
        self._make_manifest(job_config)

    def _make_manifest(self, job_config):
        self.container_volume_mount_list = \
            job_config.get('volume_mount_list',
                           [{'name': 'workspace-nvflare', 'mountPath': '/workspace/nvflare'}]
                           )
        set_list = job_config.get('set_list')
        if set_list is None:
            self.container_args_module_args_sets = list()
        else:
            self.container_args_module_args_sets = ['--set'] + set_list
        self.container_args_module_args_dict = \
            job_config.get('module_args',
                           {
                               '-m': None,
                               '-w': None,
                               '-t': None,
                               '-d': None,
                               '-n': None,
                               '-c': None,
                               '-p': None,
                               '-g': None,
                               '-scheme': None,
                               '-s': None
                           }
                           )
        self.container_args_module_args_dict_as_list = list()
        for k, v in self.container_args_module_args_dict.items():
            self.container_args_module_args_dict_as_list.append(k)
            self.container_args_module_args_dict_as_list.append(v)
        self.volume_list = \
            job_config.get('volume_list',
                           [{
                               'name': None,
                               'hostPath': {
                                   'path': None,
                                   'type': 'Directory'
                               }
                           }]
                           )

        self.pod_manifest['metadata']['name'] = job_config.get('name')
        self.pod_manifest['spec']['containers'] = self.container_list
        self.pod_manifest['spec']['volumes'] = self.volume_list

        self.container_list[0]['image'] = job_config.get('image', 'nvflare/nvflare:2.5.0')
        self.container_list[0]['name'] = job_config.get('container_name', 'nvflare_job')
        self.container_list[0]['args'] = \
            self.container_args_python_args_list + \
            self.container_args_module_args_dict_as_list + \
            self.container_args_module_args_sets
        self.container_list[0]['volumeMounts'] = self.container_volume_mount_list

    def get_manifest(self):
        return self.pod_manifest

    def abort(self, timeout=None):
        resp = self.api_instance.delete_namespaced_pod(name=self.job_id, namespace=self.namespace, grace_period_seconds=0)
        return self.enter_states([JobState.TERMINATED], timeout=timeout)

    def get_state(self):
        try:
            resp = self.api_instance.read_namespaced_pod(name=self.job_id, namespace=self.namespace)
        except ApiException as e:
            return JobState.UNKNOWN
        return POD_STATE_MAPPING.get(resp.status.phase, JobState.UNKNOWN)

    def enter_states(self, job_states_to_enter: list, timeout=None):
        starting_time = time.time()
        if not isinstance(job_states_to_enter, (list, tuple)):
            job_states_to_enter = [job_states_to_enter]
        if not all([isinstance(js, JobState)] for js in job_states_to_enter):
            raise ValueError(f"expect job_states_to_enter with valid values, but get {job_states_to_enter}")
        while True:
            job_state = self.get_state()
            if job_state in job_states_to_enter:
                return True
            elif timeout is not None and time.time()-starting_time>timeout:
                return False
            time.sleep(1)


class K8sJobLauncher(JobLauncherSpec):
    def __init__(self, config_file_path, namespace='default'):
        super().__init__()

        config.load_kube_config(config_file_path)
        try:
            c = Configuration().get_default_copy()
        except AttributeError:
            c = Configuration()
            c.assert_hostname = False
        Configuration.set_default(c)
        self.core_v1 = core_v1_api.CoreV1Api()
        self.namespace = namespace

        self.job_handle = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def launch_job(self, client, startup, job_id, args, app_custom_folder, target: str, scheme: str,
                   timeout=None) -> bool:

        root_hostpath = "/home/azureuser/wksp/k2k/disk"
        job_image = "localhost:32000/nvfl-k8s:0.0.1"
        job_config = {
            "name": job_id,
            "image": job_image,
            "container_name": f"container-{job_id}",
            "volume_mount_list": [{'name':'workspace-nvflare', 'mountPath': '/workspace/nvflare'}],
            "volume_list": [{
                'name': "workspace-nvflare",
                'hostPath': {
                    'path': root_hostpath,
                    'type': 'Directory'
                    }
            }],
            "module_args": {
                '-m': args.workspace,
                '-w': startup,
                '-t': client.token,
                '-d': client.ssid,
                '-n': job_id,
                '-c': client.client_name,
                '-p': "tcp://parent-pod:8004",
                '-g': target,
                '-scheme': scheme,
                '-s': "fed_client.json"
            },
            "set_list": args.set
        }

        self.logger.info(f"launch job with k8s_launcher. Job_id:{job_id}")

        self.job_handle = K8sJobHandle(job_id, self.core_v1, job_config, namespace=self.namespace)
        try:
            self.core_v1.create_namespaced_pod(body=self.job_handle.get_manifest(),  namespace=self.namespace)
            if self.job_handle.enter_states([JobState.RUNNING], timeout=timeout):
                return True
            else:
                return False
        except ApiException as e:
            return False

    def terminate(self):
        if self.job_handle:
            self.job_handle.abort()

    def poll(self):
        if self.job_handle:
            return self.job_handle.get_state()
        else:
            return JobState.UNKNOWN

    def wait(self):
        if self.job_handle:
            self.job_handle.enter_states([JobState.SUCCEEDED, JobState.TERMINATED])

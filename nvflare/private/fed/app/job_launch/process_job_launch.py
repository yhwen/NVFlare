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
import os
import shlex
import subprocess
import sys

from nvflare.private.fed.app.job_launch.job_launch_spec import JobLaunchSpec
from nvflare.private.fed.utils.fed_utils import add_custom_dir_to_path


class ProcessJobLaunch(JobLaunchSpec):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

        self.process = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def launch_job(self,
                   client,
                   startup,
                   job_id,
                   args,
                   app_custom_folder,
                   target: str,
                   scheme: str,
                   timeout=None) -> bool:

        new_env = os.environ.copy()
        if app_custom_folder != "":
            add_custom_dir_to_path(app_custom_folder, new_env)

        command_options = ""
        for t in args.set:
            command_options += " " + t
        command = (
            f"{sys.executable} -m nvflare.private.fed.app.client.worker_process -m "
            + args.workspace
            + " -w "
            + startup
            + " -t "
            + client.token
            + " -d "
            + client.ssid
            + " -n "
            + job_id
            + " -c "
            + client.client_name
            + " -p "
            + str(client.cell.get_internal_listener_url())
            + " -g "
            + target
            + " -scheme "
            + scheme
            + " -s fed_client.json "
            " --set" + command_options + " print_conf=True"
        )
        # use os.setsid to create new process group ID
        self.process = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, env=new_env)

        self.logger.info("Worker child process ID: {}".format(self.process.pid))

    def terminate(self):
        try:
            os.killpg(os.getpgid(self.process.pid), 9)
            self.logger.debug("kill signal sent")
        except:
            pass

        self.process.terminate()

    def return_code(self):
        if self.process:
            return self.process.poll()
        else:
            return None

    def wait(self):
        if self.process:
            self.process.wait()

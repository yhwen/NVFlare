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

import uuid

from nvflare.app_common.job.fed_app_config import ClientAppConfig, ServerAppConfig, FedAppConfig
from nvflare.app_common.job.fed_job_config import FedJobConfig
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_opt.pt import PTFileModelPersistor
from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.app_common.widgets.external_configurator import ExternalConfigurator
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from net import Net
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver

# TODO: FedApp -> server, FedApp -> client


class FedJob:
    def __init__(self, job_name="client-api-pt", workspace="/tmp/nvflare/simulator_workspace") -> None:
        self.job_name = job_name
        self.job_id = str(uuid.uuid4())
        self.workspace = workspace
        self.root_url = ""

        self.job = self.define_job()

    def define_job(self) -> FedJobConfig:
        # job = FedJobConfig(job_name="hello-pt", min_clients=2, mandatory_clients="site-1")
        job: FedJobConfig = FedJobConfig(job_name=self.job_name, min_clients=2)

        # TODO: implement in .to() call
        server_app = self._create_server_app(min_clients=2, num_rounds=2)
        app = FedAppConfig(server_app=server_app, client_app=None)
        job.add_fed_app("server", app)
        job.set_site_app("server", "server")

        client_app = self._create_client_app(site_name="site-1", app_script="cifar10_fl.py")
        app = FedAppConfig(server_app=None, client_app=client_app)
        job.add_fed_app("app1", app)
        job.set_site_app("site-1", "app1")

        client_app = self._create_client_app(site_name="site-2", app_script="cifar10_fl.py")
        app = FedAppConfig(server_app=None, client_app=client_app)
        job.add_fed_app("app2", app)
        job.set_site_app("site-2", "app2")

        return job

    def _create_client_app(self, site_name, app_script, app_config=""):
        client_app = ClientAppConfig()
        executor = PTClientAPILauncherExecutor(
            launcher_id="launcher",
            pipe_id="pipe",
            heartbeat_timeout=60,
            params_exchange_format="pytorch",
            params_transfer_type="DIFF",
            train_with_evaluation=True,
        )
        client_app.add_executor(["train"], executor)

        component = SubprocessLauncher(script=f"python3 custom/{app_script}  {app_config}", launch_once=True)
        client_app.add_component("launcher", component)

        # TODO: Use CellPipe, create CellPipe objects as part of components that require it. Automatically set root_url in CellPipe
        component = FilePipe(
            mode=Mode.PASSIVE,
            root_path=f"{self.workspace}/{self.job_id}/{site_name}"
        )
        client_app.add_component("pipe", component)

        component = FilePipe(
            mode=Mode.PASSIVE,
            root_path=f"{self.workspace}/{self.job_id}/{site_name}"
        )
        client_app.add_component("metrics_pipe", component)

        component = MetricRelay(
            pipe_id="metrics_pipe",
            event_type="fed.analytix_log_stats",
            read_interval=0.1
        )
        client_app.add_component("metric_relay", component)

        component = ExternalConfigurator(
            component_ids=["metric_relay"]
        )
        client_app.add_component("config_preparer", component)

        client_app.add_component("net", Net())  # TODO: find another way to register files that need to be included in custom folder

        return client_app

    def _create_server_app(self, min_clients, num_rounds, model_class_path="net.Net"):
        server_app = ServerAppConfig()
        controller = FedAvg(
            min_clients=min_clients,
            num_rounds=num_rounds,
            persistor_id="persistor"
        )
        server_app.add_workflow("fedavg_ctl", controller)

        controller = CrossSiteModelEval(model_locator_id="model_locator")
        server_app.add_workflow("cross_site_validate", controller)

        component = PTFileModelPersistor(
            model={"path": model_class_path}
        )

        server_app.add_component("persistor", component)

        component = PTFileModelLocator(
            pt_persistor_id="persistor"
        )
        server_app.add_component("model_locator", component)

        component = ValidationJsonGenerator()
        server_app.add_component("json_generator", component)

        component = IntimeModelSelector(
            key_metric="accuracy"
        )
        server_app.add_component("model_selector", component)

        component = TBAnalyticsReceiver(
            events=["fed.analytix_log_stats"]
        )
        server_app.add_component("receiver", component)

        server_app.add_component("net", Net())  # TODO: have another way to register needed scripts

        return server_app

    def export_job(self, job_root):
        self.job.generate_job_config(job_root)

    def simulator_run(self, job_root):
        self.job.simulator_run(job_root, self.workspace, threads=2)


if __name__ == "__main__":
    job = FedJob()

    #job.export_job("/tmp/nvflare/jobs")
    job.simulator_run("/tmp/nvflare/jobs")

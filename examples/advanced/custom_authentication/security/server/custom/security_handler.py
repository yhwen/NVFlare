# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import NotAuthenticated


class ServerCustomSecurityHandler(FLComponent):
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.RECEIVED_CLIENT_REGISTER:
            self.authenticate(fl_ctx=fl_ctx)

    def authenticate(self, fl_ctx: FLContext):
        peer_ctx: FLContext = fl_ctx.get_peer_context()
        client_name = peer_ctx.get_identity_name()
        if client_name == "site_b":
            raise NotAuthenticated("site_b not allowed to register")

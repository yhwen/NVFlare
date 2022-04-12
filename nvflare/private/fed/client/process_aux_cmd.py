# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import pickle

from nvflare.apis.fl_constant import ReservedTopic, ReturnCode
from nvflare.apis.shareable import make_reply
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader
from nvflare.apis.request_processor import RequestProcessor


class AuxRequestProcessor(RequestProcessor):
    def get_topics(self) -> [str]:
        return [ReservedTopic.AUX_COMMAND]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx

        shareable = pickle.loads(req.body)

        run_number = req.get_header(RequestHeader.RUN_NUM)
        result = engine.send_aux_command(shareable, run_number)
        if not result:
            result = make_reply(ReturnCode.EXECUTION_EXCEPTION)

        result = pickle.dumps(result)
        message = Message(topic="reply_" + req.topic, body=result)
        return message

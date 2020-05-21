# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python lag.py --dataset=celeba128 --scale=8
python lag.py --dataset=celeba128 --scale=16
python lag.py --dataset=celeba128 --scale=32
python lag.py --dataset=bedroom128 --scale=4
python lag.py --dataset=bedroom128 --scale=8
python lag.py --dataset=church_outdoor128 --scale=4
python lag.py --dataset=church_outdoor128 --scale=8

python srgan.py --dataset=celeba128 --scale=8 --adv_weight=0.01
python srgan.py --dataset=celeba128 --scale=16 --adv_weight=0.01
python srgan.py --dataset=celeba128 --scale=32 --adv_weight=0.01
python srgan.py --dataset=bedroom128 --scale=4 --adv_weight=0.01
python srgan.py --dataset=bedroom128 --scale=8 --adv_weight=0.01
python srgan.py --dataset=church_outdoor128 --scale=4 --adv_weight=0.01
python srgan.py --dataset=church_outdoor128 --scale=8 --adv_weight=0.01

python srgan2.py --dataset=celeba128 --scale=8 --adv_weight=0.01
python srgan2.py --dataset=celeba128 --scale=16 --adv_weight=0.01
python srgan2.py --dataset=celeba128 --scale=32 --adv_weight=0.01
python srgan2.py --dataset=bedroom128 --scale=4 --adv_weight=0.01
python srgan2.py --dataset=bedroom128 --scale=8 --adv_weight=0.01
python srgan2.py --dataset=church_outdoor128 --scale=4 --adv_weight=0.01
python srgan2.py --dataset=church_outdoor128 --scale=8 --adv_weight=0.01

python srgan.py --dataset=celeba128 --scale=8 --adv_weight=0.001
python srgan.py --dataset=celeba128 --scale=16 --adv_weight=0.001
python srgan.py --dataset=celeba128 --scale=32 --adv_weight=0.001
python srgan.py --dataset=bedroom128 --scale=4 --adv_weight=0.001
python srgan.py --dataset=bedroom128 --scale=8 --adv_weight=0.001
python srgan.py --dataset=church_outdoor128 --scale=4 --adv_weight=0.001
python srgan.py --dataset=church_outdoor128 --scale=8 --adv_weight=0.001

python srgan2.py --dataset=celeba128 --scale=8 --adv_weight=0.001
python srgan2.py --dataset=celeba128 --scale=16 --adv_weight=0.001
python srgan2.py --dataset=celeba128 --scale=32 --adv_weight=0.001
python srgan2.py --dataset=bedroom128 --scale=4 --adv_weight=0.001
python srgan2.py --dataset=bedroom128 --scale=8 --adv_weight=0.001
python srgan2.py --dataset=church_outdoor128 --scale=4 --adv_weight=0.001
python srgan2.py --dataset=church_outdoor128 --scale=8 --adv_weight=0.001

python cgan.py --dataset=celeba128 --scale=8
python cgan.py --dataset=celeba128 --scale=16
python cgan.py --dataset=celeba128 --scale=32
python cgan.py --dataset=bedroom128 --scale=4
python cgan.py --dataset=bedroom128 --scale=8
python cgan.py --dataset=church_outdoor128 --scale=4
python cgan.py --dataset=church_outdoor128 --scale=8

python edsr.py --dataset=celeba128 --scale=8
python edsr.py --dataset=celeba128 --scale=16
python edsr.py --dataset=celeba128 --scale=32
python edsr.py --dataset=bedroom128 --scale=4
python edsr.py --dataset=bedroom128 --scale=8
python edsr.py --dataset=church_outdoor128 --scale=4
python edsr.py --dataset=church_outdoor128 --scale=8

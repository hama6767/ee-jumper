# ROS 2 launchでシミュレータとXYZ操作を起動

ROS 2 Humble / Python 3.10で確認しています。コアのMuJoCoシミュレータはROSなしでも利用できます。

## インストール

このリポジトリをROSワークスペースの `src/ee_jumper` に配置します。
ここでは `~/dev_ws` を使います。別の環境ではパスを置き換えてください。

```bash
cd ~/dev_ws/src/ee_jumper
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[gui,dev,render]'

source /opt/ros/humble/setup.bash
cd ~/dev_ws
colcon build --packages-select ee_jumper
source install/setup.bash
ros2 launch ee_jumper swim.launch.py
```

PySide6の **EE Jumper · XYZ control · Swim lab** が1画面で開きます。
3D表示・XYZ操作・速度／姿勢／変位／パワーのグラフ・22関節の状態一覧を統合しています。
初期状態はゼロ指令です。キーを押すまで泳ぎません。
3D部分をドラッグすると視点が回転し、ホイールでズームできます。
「Tilt 25°」は機体を傾けてリセットし、浮心による復元を確認するための機能です。

| 操作 | 指令 |
|---|---|
| W / S | 機体Xの前 / 後 |
| A / D | 機体Yの左 / 右 |
| R / F | 機体Zの上 / 下 |
| Space / Stop | 全軸ゼロ、関節をホームへ戻す |
| Q / Esc / 操作画面を閉じる | シミュレータを含めて終了 |
| launchでCtrl-C | 統合デモを終了 |

指令はラッチ式です。キーを離しても継続し、Spaceで解除します。
複数軸を押すと斜め方向を指令します。泳いだ後の惰性は残ります。
標準指令は各軸 **0.08 m/s**、入力上限は **0.1 m/s** です。
指令値が実現可能な速度を超える場合は泳ぎ続けますが、目標速度を保証するものではありません。
座標は機体基準です。特に長時間泳いで姿勢が変わった場合、世界座標の軸とは一致しません。

## 起動オプション

```bash
# 速度を控えめにする
ros2 launch ee_jumper swim.launch.py speed:=0.04

# 操作画面の代わりにxtermを開く（xtermのインストールが必要）
ros2 launch ee_jumper swim.launch.py teleop:=terminal

# 自動制御用: GUIとキーボードノードなし
ros2 launch ee_jumper swim.launch.py viewer:=false teleop:=none

# 別の名前空間と学習データを使用
ros2 launch ee_jumper swim.launch.py topic_prefix:=/swimmer \
  library:=/absolute/path/to/strokes.json

# 指定時間後に終了
ros2 launch ee_jumper swim.launch.py duration:=10
```

入力は `/ee_jumper/cmd_vel` (`geometry_msgs/Twist`) です。
出力は `/ee_jumper/joint_states`、`/ee_jumper/joint_targets`、`/ee_jumper/odom`、`/ee_jumper/stroke`、
全体重心位置 `/ee_jumper/com_position` (`geometry_msgs/PointStamped`)。
`topic_prefix` を変えると入力・出力・操作画面の接続先がまとめて変わります。
統合GUIで外部指令を使う場合は **Local keyboard / buttons** をオフにしてください。
チェックを外すと最後にゼロ指令を1回送り、GUIからの定期送信を止めます。
0.5秒のコマンド途絶で泳法を停止します。GUIのSpaceボタンは外部操作中でもゼロを1回送ります。
`teleop:=none` では従来の `ee-jumper-keyboard` や外部ノードから指令を送れます。

## Pythonの選択

ビルド時にリポジトリ直下の `.venv/bin/python` があれば自動で使用します。
なければ `/usr/bin/python3` を使います。ROSのPythonと同じバージョンで仮想環境を作ってください。
MuJoCo / NumPy / Gymnasiumは、そのPythonに `pip install -e ".[gui]"` でインストールします。

起動時に変更する場合:

```bash
ros2 launch ee_jumper swim.launch.py python:=/path/to/venv/bin/python
```

ビルド時に固定する場合:

```bash
colcon build --packages-select ee_jumper \
  --cmake-args -DEE_JUMPER_PYTHON=/path/to/venv/bin/python
```

環境変数 `EE_JUMPER_PYTHON` でも起動時の既定値を指定できます。
GUIはPySide6（Essentials）を使います。OpenGLとデスクトップのDISPLAYが必要です。
UbuntuでQtのxcbプラグインが不足する場合は `sudo apt install libegl1 libgl1 libopengl0 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1`。
[Qt公式のLinux依存一覧](https://doc.qt.io/qt-6/linux-requirements.html)と、
最小Ubuntu環境で検証する[CI設定](../.github/workflows/test.yml)にも必要なパッケージを記載しています。
画面を使わない環境では `viewer:=false teleop:=none` にします。
学習データ・コードを変更したら `colcon build --packages-select ee_jumper` でインストール先も更新してください。

## 検証

実際のlaunch、統合PySide6 GUIのキー入力、ROSメッセージ、無操作時の静止、連動終了を検証します。
`python-xlib` とXvfbはテスト用で、通常の起動には不要です。

```bash
source /opt/ros/humble/setup.bash
source ~/dev_ws/install/setup.bash
cd ~/dev_ws/src/ee_jumper
.venv/bin/python -m pip install python-xlib
xvfb-run -a .venv/bin/python scripts/check_launch.py
```

実測した結果は [`examples/launch-check.json`](../examples/launch-check.json) に保存しています。

launch定義は [`launch/swim.launch.py`](../launch/swim.launch.py)、ROSパッケージ定義は
[`package.xml`](../package.xml) と [`CMakeLists.txt`](../CMakeLists.txt) です。

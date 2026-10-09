# 検証記録

このページは初回の前進デモの記録です。追加した6方向の周期遊泳とROS 2操作については
[周期遊泳の検証・操作手順](cycle_control.md)を参照してください。

2026-10-09、Linux / Python 3.10.12、MuJoCo 3.15.0、Gymnasium 1.4.0、
NumPy 2.2.6で実施。GPUは使用していません。
再現用の主要バージョンは `requirements-tested.txt` に記録しています。

## CEMの学習と評価

```bash
ee-jumper train --generations 6 --population 16 --seconds 4 \
  --seed 0 --output runs/first-gait.json
```

この出力を `examples/planar-gait.json` として同梱しています。
学習と3つの未使用seedでの評価を含む実行時間は136.6秒でした。
マシン性能とライブラリのバージョンに依存する参考値です。

| 初期状態seed | 姿勢保持だけの前進距離 | 学習した動きの前進距離 | 学習した動きの平均速度 |
|---|---:|---:|---:|
| 0（学習） | 0.000031 m | 0.06077 m | 0.01519 m/s |
| 100（評価） | −0.000073 m | 0.06064 m | 0.01516 m/s |
| 101（評価） | −0.000055 m | 0.06111 m | 0.01528 m/s |
| 102（評価） | 0.000014 m | 0.06069 m | 0.01517 m/s |

全て4秒間、planar、同一の流体条件です。わずかな初期関節角の違いに対する検証であり、
異なる流れや姿勢でも泳げることを示したものではありません。

## 短い評価の限界

4秒の結果には開始直後の過渡運動が含まれます。
同じ方策をseed=100で長く再生した結果は以下です。

| 時間 | 全体重心の前進距離 |
|---:|---:|
| 5 s | 0.05622 m |
| 10 s | 0.06674 m |
| 15 s | 0.07674 m |
| 20 s | 0.08733 m |
| 25 s | 0.09752 m |
| 30 s | 0.10814 m |

30秒平均は0.00360 m/s、5–30秒の区間平均は約0.00208 m/sです。
**初動後もゆっくり前進しますが、持続速度の改善はこれからです。**
次の実験では `--seconds 20` などで探索期間を延ばし、初期の過渡運動を除いた評価や
複数周期にわたる移動距離を学習の基準にすることを推奨します。

planarで学習した同じ方策をfreeで4秒再生すると、前進0.05998 m、
深度誤差−0.00734 m、胴体上方向と鉛直の内積0.9568でした。
これは追加の短い動作確認で、3D姿勢制御の学習・収束・長時間安定性の実証ではありません。
数値原本は `examples/extended-evaluation.json` です。

## 自動テスト

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`: **13 passed**。

- 取り込んだ41メッシュと変更後XMLのSHA-256。
- 22関節、モデル質量、planar/free双方のGymnasium契約、seed再現性、時間上限。
- 中性浮力と重力のつり合い、浮力不足による沈降。
- 抵抗が並進運動を妨げること、流体と等速なら流体力が消えること。
- 斜め流れに対して揚力係数の変更が力に反映されること。
- 流体・重力なしで内部の関節運動だけでは全体重心が前進しないこと。
- ランダム入力の有限性、トルク上限、不正入力、数値破綻時の有限な終了通知。
- 小さなCEM探索の保存・読み込み・再評価。

Gymnasiumは観測空間の±無限大の上限に警告を出します。速度などの真値を返す
連続状態空間を宣言しているためで、観測値そのものが無限大になるという意味ではありません。

`ruff check src tests scripts` に合格。EGLで画像と4秒のGIFを描画し、画像を確認済みです。
学習済み方策の静止画を `docs/media/swim.png` に掲載しています。
宣言した下限バージョンのMuJoCo 3.3.0でも、同じ13テストに合格しました。

## PPO

PyTorch 2.14.1+cpu / Stable-Baselines3 2.9.0で次を実行しました。

```bash
python scripts/train_ppo.py --steps 512 --envs 1 --output runs/ppo-smoke
```

512ステップの収集、PPO更新、保存、再読み込み、10ステップの決定的推論を確認しました。
PPOの泳ぎ方の収束は検証していません。学習済みデモはCEMのJSONです。

## 配布と公開状態

Apache-2.0のLICENSE、派生元のNOTICE、コミット固定の取り込みスクリプト、
SHA-256付きのprovenance、CI定義を用意しました。
`python -m build` でsdistとwheelを生成し、wheelに41メッシュとLICENSE/NOTICEが入ること、
ソースツリー外にインストールしてplanar/freeを動かせることを確認しています。
GitHubへのリポジトリ作成・push、PyPIへの公開は行っていません。


## 中性浮力修正と速度優先の再学習（浮心オフセット導入前）

描画同期が `xfrc_applied` をクリアして浮力を失う不具合を修正しました。
浮力は各リンクの `gravcomp` でモデル内に設定し、推力計測からはこの成分を差し引きます。
水平・傾斜姿勢でビューアを同期しながら各30秒間、位置・姿勢・関節角が数値誤差内で静止することを確認しました。
GUI終了時のレンダースレッドとGLFW終了の競合も修正しました。

現行の[中性浮力の検証記録](../examples/neutral-buoyancy.json)は下記の受動復元テストに更新しています。

[速度と機械パワーの旧新比較](../examples/speed-comparison.json)、
[6方向の自由遊泳・細かい刻み幅での検証](../examples/strokes-validation.json)、
[ROS 2＋ビューアの検証](../examples/ros-check.json)を保存しています。
最新の操作方法・評価条件は[周期遊泳の説明](cycle_control.md)を参照してください。


## 大振幅・正味変位の学習とROS 2 launch

v2の6方向ライブラリを自由遊泳の6秒変位から再学習しました。
物理モデルの質量・関節範囲・2 N·mのトルク上限・中性浮力は維持し、
探索振幅の上限を1.6 rad、周期を1秒に変更しています。
[旧新比較](../examples/displacement-comparison.json)では全方向で正味変位が増え、
前回比1.98〜4.84倍になりました。角度の実測幅も比較記録に保存しています。

コア32テスト、細かい時間刻みでの再評価、10秒の速度指令を検証しています。
GUI付きの実際の `ros2 launch ee_jumper swim.launch.py` について、
両画面の起動、無操作での静止、WキーによるROS指令・遊泳、Space停止、Qによる両ノードの正常終了を確認しました。
[launch検証記録](../examples/launch-check.json)と[起動方法](ros_launch.md)を参照してください。


## PySide6と浮心オフセット

現在の標準モデルは浮心を重心から機体+Zへ10 mmずらしています。
各リンクの中性浮力を維持し、`step_water`で各物理ステップの復元偶力を計算します。
以前の「傾斜姿勢でも静止する」仕様から「水平は静止し、傾斜は受動的に復元する」仕様へ変更しました。

- コア34テストにMuJoCo 3.15.0 / 3.3.0の両方で合格。
- 水平30秒のドリフトは数値誤差内。25°のroll / pitchから30秒後の傾きは約0.04°。
- 6泳法を再計測。全方向で採用条件を満たし、最大姿勢変化が減少。
- 物理刻み0.5 ms・制御5 msでも自由遊泳と推力計を確認。
- PySide6上で静止、傾斜復元、Wキーでの前進、Space、リセット、CSV出力、Q終了を確認。
- シミュレーションはGUIとROSで `simulation.Swimmer` を共用。端末操作は小さなROSアダプタとして維持。

原本: [浮力](../examples/neutral-buoyancy.json)、[泳法比較](../examples/passive-swimming-comparison.json)、
[細かい時間刻み・閉ループ](../examples/strokes-validation.json)、
[統合GUI](../examples/dashboard-check.json)、[ROS launch](../examples/launch-check.json)。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python scripts/check_neutral_buoyancy.py
python scripts/validate_cycles.py --library examples/strokes-xyz.json
xvfb-run -a python scripts/check_dashboard.py
# ROSワークスペースをビルド・sourceした後
xvfb-run -a python scripts/check_launch.py
```

GUI検証はPython 3.10 / PySide6-Essentials 6.8.3。描画は800×450・20 Hz、
高負荷な影・反射・マルチサンプリングを省き、Xvfbでも約1倍速で動作しました。
`ros2 launch`からのW指令4秒で約27.7 cm前進し、Space / Qによる停止・終了も確認しました。

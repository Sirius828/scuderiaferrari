# 少参数巡线控制器

控制器只接受完整的 `/perception/lane_state` 帧。控制量固定为近场误差 PD 加一项弯道前馈，速度只分 `STRAIGHT` 和 `CURVE` 两态。

日常只调以下 7 个参数：

| 现象 | 只调整 | 建议步长 |
|---|---|---|
| 直道摆动或回正过冲 | `Kp`、`Kd` | `Kp +0.05`，`Kd +0.01` |
| 入弯迟钝、弯中向外或卸舵 | `curve_feedforward_gain` | `+0.05` |
| 速度状态过早或过晚切换 | `curve_enter_threshold`、`curve_exit_threshold` | 每次只改一个 |
| 姿态正确但平均速度低 | `curve_speed`、`straight_speed` | 弯道 `+0.05 m/s`，直道 `+0.10 m/s` |

第一阶段可用 `config/controller_params_pd_baseline.yaml` 关闭前馈，以全程 `0.6 m/s` 调 PD。恢复默认配置后再调前馈和阈值。

诊断工具绕过 `ros2 topic` 命令，直接使用 rclpy：

```bash
python3 tools/line_follow_diag.py --apply-defaults --watch 60 --record /tmp/laps.csv
python3 tools/line_follow_diag.py --kp 0.85 --kd 0.06 --watch 60
python3 tools/line_follow_diag.py --curve-feedforward-gain 0.30 --watch 60
python3 tools/line_follow_diag.py --curve-speed 0.70 --straight-speed 0.85 --watch 60
```

工具每秒输出近场误差 P95、舵量饱和率、P/D/FF 峰值和平均命令速度，并以每两次入弯估算一圈。CSV 每个控制帧一行，包含拟合源过渡状态。

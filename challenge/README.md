
## DATA format

The data format is [NDJSON](https://github.com/ndjson/ndjson-spec): one full 360-degree turn per line.

```json
[epochMs, [[distanceMm, intensity, angleQ6], ...]]
```

- `epochMs`: turn start time in Unix epoch milliseconds
- `distanceMm`: distance in millimeters, or `0` for no return
- `intensity`: reported sample quality
- `angleQ6`: angle in `degrees * 64`


i Jason - That was quick. Friday works for us, and will setup our meeting. Regarding preparation, we are in the people and asset tracking space, using AI on sensors from devices to get accurate x,y, and z location. As part of the interview process, we have prepared a 2D LIDAR dataset for candidates to work on. See the link here: ​gd_challenge​. You wont get much time to solve the challenge before Friday, but if you can, all good. Classic ICP techniques may not work here, FYI. However, if a student can propose a thoughtful and workable approach, and get the output close to the expected trajectory, that’s exactly what we want to see. This is a one‑week challenge. Brief description of the LiDAR dataset: The dataset includes two components: A README file describing the structure and contents. An output.png file showing the reference trajectory. The task is to take the provided .json LiDAR data and develop algorithms that reconstruct the trajectory shown in output.png. The reference plot is included so students can understand the limitations of relying solely on AI‑generated solutions and instead work toward approximating the trajectory through their own reasoning and experimentation. Achieving a high‑quality reconstruction will naturally take time, which is part of the intended learning experience. Best, Devu heres the email and the two files they gave me and a sample of the lidar data since its pre processing
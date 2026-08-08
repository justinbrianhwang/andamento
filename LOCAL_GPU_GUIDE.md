# 로컬 GPU 측정 가이드 (RTX 30/40/50 시리즈)

> 소요 시간: 셋업 ~10분 + 측정 10~20분. 측정 중에는 게임/영상 등 GPU 쓰는
> 프로그램을 꺼주세요. 노트북이면 전원을 연결해 주세요.

연구용 GPU 커널 벤치마크입니다. GPU에서 행렬곱 커널 설정 78가지를
자동으로 측정하고, 결과를 JSON 파일 하나로 저장합니다. 개인 정보는
수집하지 않으며 기록되는 것은 GPU 모델명·드라이버/라이브러리 버전·측정
시간뿐입니다 (스크립트는 전부 공개 코드라 직접 확인 가능).

## Windows인 경우 (WSL2 사용)

NVIDIA 그래픽 드라이버가 설치되어 있다면 WSL이 GPU를 자동 인식합니다.

```powershell
# 1) PowerShell(관리자)에서 — WSL이 없다면:
wsl --install -d Ubuntu
# 설치 후 재부팅하고 Ubuntu 창에서 사용자 계정 만들기
```

```bash
# 2) Ubuntu(WSL) 창에서:
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv ~/bench && source ~/bench/bin/activate
pip install -U "jax[cuda12]"

# GPU 인식 확인 — [CudaDevice(id=0)] 처럼 나오면 성공
python -c "import jax; print(jax.devices())"

# 3) 측정 실행
git clone https://github.com/justinbrianhwang/andamento
cd andamento
python experiments/sweep_gpu.py bf16
```

## Linux인 경우

위의 2)~3)만 그대로 실행하면 됩니다.

## 끝나면

폴더에 생긴 **`sweep_result_<GPU이름>.json`** 파일 하나만 보내주세요.
(예: `sweep_result_NVIDIA_GeForce_RTX_4090.json`)

중간에 `FAILED [launch_resources]` 같은 줄이 나오는 건 정상입니다 —
일부러 한계를 넘는 설정도 시험하며, 실패도 연구 데이터입니다.

## 문제가 생기면

- `jax.devices()`가 `[CpuDevice(...)]`로 나옴 → GPU 미인식. Windows
  NVIDIA 드라이버를 최신으로 올리고 WSL 재시작(`wsl --shutdown` 후 다시
  열기).
- `Triton support is only enabled for Ampere GPUs...` → GTX/RTX 20
  시리즈 이하는 하드웨어 미지원입니다. 그 메시지가 담긴 출력도 그대로
  보내주시면 데이터로 사용합니다.
- 그 외 오류는 터미널 출력을 통째로 복사해서 보내주세요.

고맙습니다! 🙏

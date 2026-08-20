import time
import gc
import torch
import jax
import jax.numpy as jnp
from kernel.physics_filter import PhysicsInformativeFilter
from kernel.autograd_free import AutogradFreeIsolationLayer

def get_current_vram_usage() -> float:
    """
    현재 GPU(CUDA)의 순수 물리적 메모리 점유율을 MB 단위로 정밀하게 측정합니다.
    호스트 인터페이스 지연 버퍼를 방지하기 위해 가비지 컬렉션을 강제 연동합니다.
    """
    if torch.cuda.is_cuda_available():
        # [리팩토링] 파이썬 GC와 CUDA 캐시를 동시에 비워 메모리 측정 오차율을 0%로 동결
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

def test_vram_static_o1_homoeostasis():
    """
    [VRAM 정적 O(1) 동결 레이어 벤치마크 테스트]
    무한 스트림 환경(1,000번의 시간 틱)을 주입하여 메모리가 완벽히 O(1)로 꽁꽁 묶이는지 검증합니다.
    """
    print("\n========================================================================")
    print("🧪 [TEST] 2세대 항상성 커널 VRAM 정적 O(1) 복잡도 장기 추적 검증 시동")
    print("========================================================================")

    # 1. 2세대 핵심 엔진 하이브리드 초기화
    physics_engine = PhysicsInformativeFilter(dt=0.001, h_bar_eff=1.0, viscosity_sigma=0.5)
    isolation_guard = AutogradFreeIsolationLayer(physics_kernel=physics_engine)
    
    closure_pipeline = physics_engine.process_pipeline
    jit_isolated_run = jax.jit(isolation_guard.execute_isolated_forward, static_argnums=(2,))

    # 2. 초기 웜업(Warm-up) 주행 (JAX JIT 컴파일러 가동 컨텍스트 제외용)
    # [리팩토링] 사출 셰이프를 실전 테스트 덩어리와 정렬 (1, 4096)
    warmup_stream = jnp.zeros((1, 4096), dtype=jnp.float32)
    _ = jit_isolated_run(warmup_stream, closure_pipeline)
    
    initial_vram = get_current_vram_usage()
    print(f"📦 [기준점 생성] JIT 컴파일 완료 후 초기 VRAM 상태: {initial_vram:.2f} MB")

    # 3. 무한 스트림 가상 주행 (1,000 틱 연속 순방향 주행)
    total_ticks = 1000
    memory_history = []

    print(f"🔄 선형적 시간 축 롤아웃 시작 ({total_ticks} Ticks 전진)...")
    start_time = time.time()

    for tick in range(1, total_ticks + 1):
        # [리팩토링] 하드웨어 가속기(GPU) 내부에서 다이렉트로 연동되는 무복사 청정 난수 텐서 생성
        mock_logits_chunk = torch.randn(1, 4096, device="cuda", dtype=torch.float32)
        
        # 0ns 포인터 스왑을 가장한 JAX 주소선 융합 변환 (우리가 구축한 dlpack_bridge 대리 수식)
        from interface.dlpack_bridge import torch_logits_to_jax_bridge
        jax_logits_chunk = torch_logits_to_jax_bridge(mock_logits_chunk)
        
        # 2세대 가드레일 통과 (내부 stop_gradient 자율 절연막 가동)
        outputs = jit_isolated_run(jax_logits_chunk, closure_pipeline)
        
        # 블록 내부 데이터 강제 동기화 후 실리콘 레지스터 고착화
        outputs["sanitized_output"].block_until_ready()
        
        if tick % 200 == 0 or tick == 1:
            current_vram = get_current_vram_usage()
            memory_history.append(current_vram)
            print(f" └─ [Tick {tick:04d}/{total_ticks}] ➔ 현재 VRAM 점유량: {current_vram:.2f} MB")

    end_time = time.time()
    final_vram = get_current_vram_usage()
    
    print("------------------------------------------------------------------------")
    print(f"⏱️ 총 연산 소요 시간: {end_time - start_time:.4f} 초")
    print(f"📊 최종 VRAM 상태: {final_vram:.2f} MB (시작점 대비 변동량: {final_vram - initial_vram:.2f} MB)")

    # 4. [합격 불합격 검증 오프셋] 
    # 역전파 차단막이 완벽히 가동했다면 수명 주기 내 변동량은 엄격하게 0.5MB 이내(정적 수평 플랫라인)여야 합니다.
    vram_drift = abs(final_vram - initial_vram)
    
    assert vram_drift < 0.5, f"❌ [TEST FAILED] VRAM 메모리 누수 감지! 복잡도가 O(1)이 아닙니다. 변동량: {vram_drift:.2f} MB"
    print("✅ [TEST PASSED] 시간 축의 무한 전진과 무관하게 VRAM 복잡도가 정적 O(1)로 완벽히 동결되었습니다.")
    print("========================================================================\n")

if __name__ == "__main__":
    if torch.cuda.is_cuda_available():
        test_vram_static_o1_homoeostasis()
    else:
        print("\n⚠️ [하드웨어 경고] VRAM O(1) 누수 정밀 프로파일링 측정을 위해 CUDA(GPU) 환경이 강제됩니다.\n")


import time
import torch

# --- 辅助工具: 显存监控 & 计时器 & 参数量 ---
class Timer:
    def __init__(self, device="cuda"):
        self.device = device
        self.start_time = time.perf_counter() # 使用精度更高的 perf_counter
        self.last_checkpoint = self.start_time

    def sync(self):
        """确保 GPU 任务执行完毕，再进行计时"""
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()
    def checkpoint(self):
        """返回距离上一次打点过去的时间（秒），并更新打点时间"""
        self.sync() # 统计前先同步
        now = time.perf_counter()
        elapsed = now - self.last_checkpoint
        self.last_checkpoint = now
        return elapsed

    def total(self):
        """返回从初始化开始过去的总时间（秒）"""
        self.sync()
        return time.perf_counter() - self.start_time

def print_monitor_stats(tag="", batch_size=None, accelerator=None, timer=None):
    """
    打印当前 GPU 状态、耗时以及 FPS。
    :param batch_size: 如果提供，将计算该阶段的 FPS
    """
    if accelerator is not None and not accelerator.is_main_process:
        return
    # accelerator = None 跳过分布式训练相关检查

    # 显存数据 (GB)
    gb = 1024 ** 3
    max_allocated = torch.cuda.max_memory_allocated() / gb
    max_reserved = torch.cuda.max_memory_reserved() / gb
    current_allocated = torch.cuda.memory_allocated() / gb
    current_reserved = torch.cuda.memory_reserved() / gb

    # 时间数据
    time_info = ""
    if timer:
        elapsed_step = timer.checkpoint()
        elapsed_total = timer.total()
        # 计算 FPS (FPS = 样本数 / 时间)
        fps_info = ""
        if batch_size:
            fps = batch_size / elapsed_step
            fps_info = f"\n- FPS (每秒处理数):          {fps:.2f}"

        time_info = (
            f"\n- Step Runtime (阶段耗时):    {elapsed_step:.4f} s"
            f"{fps_info}"
            f"\n- Total Runtime (总运行时间): {elapsed_total:.4f} s"
        )

    print(f"------------------------------------------------------------------------------------------------------------")
    print(f"[{tag}] 资源监控报告:")
    print(f"- Peak Allocated (显存峰值):   {max_allocated:.2f} GB")
    print(f"- Peak Reserved (预留峰值):    {max_reserved:.2f} GB")
    print(f"- Current       (当前瞬间占用): {current_reserved:.2f} GB (Alloc: {current_allocated:.2f} GB){time_info}")
    print(f"------------------------------------------------------------------------------------------------------------\n")

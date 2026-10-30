from app.pipeline.device import DeviceManager


def test_device_detection_and_verification():
    mgr = DeviceManager()
    info = mgr.get_device_info()
    assert info["device"] in ("mps", "cuda", "cpu")
    assert info["verified"] is True
    assert "device_name" in info


def test_device_forced_cpu():
    mgr = DeviceManager(preferred_device="cpu")
    assert mgr.device == "cpu"
    assert "CPU" in mgr.device_name

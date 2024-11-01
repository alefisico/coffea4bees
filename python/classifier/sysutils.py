def recursive_interrupt():
    import psutil
    import signal

    p = psutil.Process()
    for child in p.children(recursive=True):
        child.send_signal(signal.SIGINT)

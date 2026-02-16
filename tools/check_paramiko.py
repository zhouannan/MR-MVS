try:
    import paramiko
    print(paramiko.__version__)
except Exception as e:
    print("ERR:", e)

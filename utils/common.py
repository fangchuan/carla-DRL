# DEBUG_PRINT函数，去除print时只需要将const.DEBUG=0
try:
    import const
except:
    from . import const

const.DEBUG = 1

def DEBUG_PRINT(*kwargs):
    if const.DEBUG:
        print(*kwargs)

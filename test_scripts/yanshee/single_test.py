import YanAPI
import time
from turn_and_translate import tat
from translate_test import translate

def task2():
    YanAPI.sync_play_motion("A_start_all_908A")


    YanAPI.sync_play_motion("come_up_09712")
    YanAPI.sync_play_motion("restoration_h_04")
    YanAPI.sync_play_motion("come_up_09712")
    YanAPI.sync_play_motion("come_down_0908")
    YanAPI.sync_play_motion("restoration_h_04")
    YanAPI.sync_play_motion("come_down_0908A")





















if __name__ == "__main__":
    task2()
import YanAPI
import time
from turn_and_translate import tat
from translate_test import translate

def task2():
    # #
    # # # 任务二————————————————————————————————————————
    # #
    # # # 启动区------------------------------------------
    YanAPI.sync_play_motion("A_start_all_908A")


    # 台阶--------------------------------------------
    YanAPI.sync_play_motion("come_up_09712")
    YanAPI.sync_play_motion("restoration_h_04")
    YanAPI.sync_play_motion("come_up_09712")
    YanAPI.sync_play_motion("come_down_0908")
    YanAPI.sync_play_motion("restoration_h_04")
    YanAPI.sync_play_motion("come_down_0908A")

    # # #
    # # # #
    # # # # #跨栏1分21秒变1分15秒------------------------------------------
    #q4
    # YanAPI.sync_play_motion("A_start_908B")

    #1
    # YanAPI.sync_play_motion("A_start_all_ts2")

    #2
    # YanAPI.sync_play_motion("A_start_all_ts2")
    # YanAPI.sync_play_motion("A_walk_zhu_ts2")

    #3
    # YanAPI.sync_play_motion("A_start_all_ts2")
    # YanAPI.sync_play_motion("A_walk_zhu_ts1")

    #4
    # YanAPI.sync_play_motion("A_start_all_ts3")

    #5
    # YanAPI.sync_play_motion("A_start_all_ts3")
    # YanAPI.sync_play_motion("A_walk_zhu_ts1")

    #6
    # YanAPI.sync_play_motion("A_start_all_ts3")
    # YanAPI.sync_play_motion("A_walk_zhu_ts1")


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #kl
    # YanAPI.sync_play_motion("hurdles_807")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #前进
    # YanAPI.sync_play_motion("restoration_h_05")


#==================================================================================================
    #平移

    # YanAPI.sync_play_motion("trte_L_J_big_18b")

    #2
    # YanAPI.sync_play_motion("trte_L_J_big_18b")
    # YanAPI.sync_play_motion("te_left_1_805")


#4444444444444444444444￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥
    #前进1
    # YanAPI.sync_play_motion("A_start_all_ts2")

    #2
    # YanAPI.sync_play_motion("A_start_all_ts2")
    # YanAPI.sync_play_motion("A_walk_zhu_ts2")

    #3
    # YanAPI.sync_play_motion("A_start_all_ts2")
    # YanAPI.sync_play_motion("A_walk_zhu_ts1")

    #4
    # YanAPI.sync_play_motion("A_start_all_ts2")
    # YanAPI.sync_play_motion("A_walk_zhu_ts2")


if __name__ == "__main__":
    task2()
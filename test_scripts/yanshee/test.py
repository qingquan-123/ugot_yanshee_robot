import YanAPI
import time
from turn_and_translate import tat
from translate_test import translate

def task2():
    YanAPI.sync_play_motion("reation_b_h_01")
    YanAPI.sync_play_motion("trte_L_J_big_18t")
    YanAPI.sync_play_motion("te_left_1_01")
    YanAPI.sync_play_motion("A_walk_01_e")
    YanAPI.sync_play_motion("trte_L_J_big_02")
    YanAPI.sync_play_motion("te_left_1_01")
    YanAPI.sync_play_motion("A_walk_01_e")
    YanAPI.sync_play_motion("clamp_622")
    YanAPI.sync_play_motion("clamp_back_805")
    YanAPI.sync_play_motion("trte_R_J_big_03")

    # red（合格）1分10秒变1分08秒
    YanAPI.sync_play_motion("turn_r_90_806A")



    tat(target_id=3)
    YanAPI.sync_play_motion("A_walk_J_02_b")
    YanAPI.sync_play_motion("A_walk_J_02_b")
    YanAPI.sync_play_motion("A_walk_J_02_b")
    YanAPI.sync_play_motion("let_go")

    # blue go B
    YanAPI.sync_play_motion("trte_R_J_big_04t")
    translate(target_id=2)
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("turn_L_90_805C")
    YanAPI.sync_play_motion("translate_r_KL3A")
    YanAPI.sync_play_motion("A_walk_zhu_02")
    YanAPI.sync_play_motion("trte_L_J_big_617")
    YanAPI.sync_play_motion("trte_L_J_big_18b")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("squat_805")
    YanAPI.sync_play_motion("Attention_805")
    YanAPI.sync_play_motion("A_walk_zhu_02")
    YanAPI.sync_play_motion("clamp_622")
    YanAPI.sync_play_motion("clamp_back_805")
    YanAPI.sync_play_motion("trte_R_J_big_03")

    # B go red
    YanAPI.sync_play_motion("turn_r_90_806A")
    tat(target_id=1)
    YanAPI.sync_play_motion("A_walk_J_02_b")
    YanAPI.sync_play_motion("A_walk_J_02_b")
    YanAPI.sync_play_motion("A_walk_J_02_b")
    YanAPI.sync_play_motion("let_go")


if __name__ == "__main__":
    task2()
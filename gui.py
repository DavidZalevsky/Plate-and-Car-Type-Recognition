from dearpygui.core import *
from dearpygui.simple import *
import time

def gui(info, visit_count, time_arrival, plate):
    set_main_window_size(500,500)
    set_style_window_menu_button_position(0)
    set_theme("Red")
    set_global_font_scale(1)
    set_main_window_resizable(False)

    welcome = "Wykryto probe wjazdu pojazdu: " + plate
    visits_count = "To twoja: " + str(visit_count) + " wizyta na tym parkingu!"
    last_visit = "Ostatnia wizyta odbyla sie:" + time.ctime(time_arrival)


    #Creates the DearPyGui Window
    with window("Informacje systemu parkingowego.",width = 500,height = 500, no_move = True):
        set_window_pos("Informacje systemu parkingowego.",0,0)
        add_logger("Witamy!",log_level=0,auto_scroll_button=False,copy_button=False,clear_button=False,filter=False)
        log_info(welcome,logger="Witamy!")
        if info == False:
            log_info(f'Wjazd do parkingu niedozwolony!.', logger="Witamy!")
        else:
            log_info(visits_count, logger="Witamy!")
            log_info(last_visit, logger="Witamy!")
            log_info("Milego pobytu!", logger="Witamy!")

    start_dearpygui()

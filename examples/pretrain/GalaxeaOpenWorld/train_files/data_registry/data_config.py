"""Galaxea Open World R1 Lite stage1 registration for QwenPI + gr00t_sharded.

All 219 subsets share one ``meta/modality.json``. Disk actions are 41-D and
absolute. Training rewrites ``left_eef`` and ``right_eef`` relative to the
chunk-start state. Grippers, arm joints, ``torso_twist``, and
``chassis_velocity`` stay absolute.

``head_right`` is present on disk and is not used.
``Turn_On_And_Off_The_Air_Conditioner_20250802_012`` has no stage1 dataset
and is omitted.

Paths are relative to ``data_root_dir``
``/mnt/zhanghao/Galaxea-Open-World-Dataset/_processed``.
"""

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class GalaxeaR1LiteStage1DataConfig:
    """Three-camera R1 Lite layout: 41-D action, relative dual EEF."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.front",
        "video.left_wrist",
        "video.right_wrist",
    ]
    # Only the reference poses for relative EEF. include_state stays false.
    state_keys = [
        "state.left_eef",
        "state.right_eef",
    ]
    action_keys = [
        "action.left_eef",
        "action.left_gripper",
        "action.right_eef",
        "action.right_gripper",
        "action.left_arm_joints",
        "action.right_arm_joints",
        "action.torso_twist",
        "action.chassis_velocity",
    ]
    state_key_dims = {
        "state.left_eef": 9,
        "state.right_eef": 9,
    }
    action_key_dims = {
        "action.left_eef": 9,
        "action.left_gripper": 1,
        "action.right_eef": 9,
        "action.right_gripper": 1,
        "action.left_arm_joints": 6,
        "action.right_arm_joints": 6,
        "action.torso_twist": 6,
        "action.chassis_velocity": 3,
    }
    # Parallel to action_keys. Consumed only by gr00t_sharded. Keys are unprefixed
    # because extract_step_data stores groups as "left_eef", not "action.left_eef".
    # Both EEFs are absolute xyz+rot6d on disk. torso_twist is 6-D and does not
    # match 3-D state.torso. chassis_velocity has no matching state.
    action_configs = [
        ActionConfig(
            rep=ActionRepresentation.RELATIVE,
            type=ActionType.EEF,
            format=ActionFormat.XYZ_ROT6D,
            state_key="left_eef",
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
            state_key="left_gripper",
        ),
        ActionConfig(
            rep=ActionRepresentation.RELATIVE,
            type=ActionType.EEF,
            format=ActionFormat.XYZ_ROT6D,
            state_key="right_eef",
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
            state_key="right_gripper",
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
            state_key="left_arm_joints",
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
            state_key="right_arm_joints",
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        # Unused by gr00t_sharded (StarVLAPackProcessor + use_percentiles).
        # Kept so the same robot_type can switch back to lerobot_datasets.
        state_modes = {key: "q99" for key in self.state_keys}
        action_modes = {key: "q99" for key in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_modes,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                ),
            ]
        )


_ROBOT_TYPE = "galaxea_r1lite_stage1"

ROBOT_TYPE_CONFIG_MAP = {
    _ROBOT_TYPE: GalaxeaR1LiteStage1DataConfig(),
}

_SUBSETS = (
    "Adjust_The_Air_Conditioner_Temperature_20250711_006",
    "Arrange_Mineral_Water_Bottles_On_The_Desk_20250721_008",
    "Arrange_Sofa_Cushions_20250708_006",
    "Arrange_Sofa_Cushions_20250804_012",
    "Arrange_The_Fruits_20250716_006",
    "Arrange_The_Fruits_20250731_012",
    "Arrange_The_Fruits_20250822_013",
    "Arrange_The_Sofa_Cushions_20250722_008",
    "Arrange_The_Tray_20250725_010",
    "Arrange_The_Tray_20250804_012",
    "Arrange_The_Tray_20250822_013",
    "Arrange_Throw_Pillows_On_Living_Room_Sofa_20250624_002",
    "Arrange_Toiletries_20250731_012",
    "Bedside_Console_With_Switch_Panel20250624_002",
    "Boil_The_Water_20250714_006",
    "Boil_The_Water_20250801_012",
    "Chair_Push_And_Place_20250709_006",
    "Chair_Push_And_Place_20250719_008",
    "Chair_Push_And_Place_20250801_012",
    "Check_Out_At_Self-checkout_Counter_20250607_003",
    "Clean_Cutting_Board_20250627_002",
    "Clean_The_Chopping_Board_20250722_008",
    "Clean_The_Cutting_Board_20250711_006",
    "Clean_The_Cutting_Board_20250804_012",
    "Clean_The_Mirror_20250714_006",
    "Clean_The_Mirror_20250731_012",
    "Clean_The_Sink_20250710_006",
    "Clean_The_Sink_20250721_008",
    "Clean_The_Sink_20250730_012",
    "Clean_The_Sofa20250627_002",
    "Clean_The_Toilet20250618_001",
    "Clean_Toilet_20250625_002",
    "Clothes_Into_The_Basket_20250728_010",
    "Collect_Juice_20250621_001",
    "Connect_Router_Cables_20250625_002",
    "Cooking_Food_In_An_Air_Fryer_20250719_008",
    "Cooking_Food_In_An_Air_Fryer_20250801_012",
    "Desktop_Garbage_Decluttering_20250712_006",
    "Desktop_Garbage_Organizing20250721_008",
    "Desktop_Trash_Disposal_20250729_011",
    "Dispose_Of_Garbage_In_The_Trash_Can",
    "Dispose_Of_Leftover_Food_20250626_002",
    "Doll_Storage_20250731_012",
    "Dry_Clothes_In_A_Dryer_20250619_001",
    "Egg_Placement20250703_002",
    "Enter_Exit_Door_Turn_On_Off_lights_20250620_001",
    "Floor_cloth_wiping_stains_20250614_001",
    "Flowers_Are_Inserted_Into_Vases_20250725_010",
    "Fold_Clothes20250617_001",
    "From_Kitchen_To_Bedroom20250703_002",
    "Handle_Plates_20250619_001",
    "Handle_The_Wooden_Barrel_Lid20250619_001",
    "Hang_Clothes20250625_002",
    "Hang_The_Towel_On_The_Shelf_20250715_006",
    "Hang_The_Towel_On_The_Shelf_20250805_012",
    "Heat_Food_In_Microwave_Oven_20250703_003",
    "Heat_Food_In_The_Microwave_20250710_006",
    "Insert_Flowers_Into_A_Vase_20250711_006",
    "Iron_The_Clothes_20250728_010",
    "Layered_Shoe_Cabinet_For_Placing_Slippers_20250728_010",
    "Lift_The_Steamer_Lid20250618_001",
    "Lift_up_the_carpet_and_wipe_the_floor_20250625_002",
    "Locker_torage_And_Retrieval20250623_002",
    "Make_Breakfast_20250626_002",
    "Make_Breakfast_20250726_010",
    "Make_Milk_Powder_20250728_010",
    "Make_Milk_Powder_20250731_012",
    "Make_Tea_20250624_002",
    "Make_The_Bed_20250616_001",
    "Make_The_Bed_20250708_006",
    "Make_The_Bed_20250730_012",
    "Milk_Powder_Preparation20250701_002",
    "Open_And_Close_Curtains_20250708_006",
    "Open_And_Close_Microwave_Oven_20250623_002",
    "Open_And_Close_Nightstand_Door_20250625_002",
    "Open_And_Close_Nightstand_Drawer_20250624_002",
    "Open_And_Close_The_Curtain_20250728_010",
    "Open_And_Close_The_Curtain_20250804_012",
    "Open_And_Close_The_Door_20250802_012",
    "Open_And_Close_The_Nightstand_To_Place_Items_20250802_012",
    "Open_And_Close_The_Notebook20250628_002",
    "Open_And_Close_The_Refrigerator_Drawer_To_Place_Items_20250728_010",
    "Open_And_Close_The_Refrigerator_Drawer_To_Place_Items_20250802_012",
    "Open_Close_Lid_Stirring_20250618_001",
    "Open_Or_Close_The_Cabinet_Door_To_Place_Items_20250719_008",
    "Open_Or_Close_The_Cabinet_Door_To_Place_Items_20250801_012",
    "Open_The_Bedroom_Door_To_Enter20250701_002",
    "Open_and_close_the_freezer_door_20250620",
    "Opening_Or_Closing_Refrigerator_Drawers_To_Store_And_Retrieve_Items20250624_002",
    "Organize_Dried_Fruit_And_Meat_Shelves_20250702_003",
    "Organize_Mineral_Water_Bottles_0n_The_Table_20250709_006",
    "Organize_Potato_Chip_Shelves_20250701_003",
    "Organize_Refrigerated_Beverage_Case_20250704_003",
    "Organize_Refrigerated_shelf_20250703_003",
    "Organize_Refrigerator_Items0250703_002",
    "Organize_The_Beverage_Cooler_20250630_003",
    "Organize_The_Bread_Shelf_20250630_003",
    "Organize_The_Coat_Rack20250617_001",
    "Organize_The_Condiment_Shelf_20250731_012",
    "Organize_The_Potato_Chip_Shelf_20250630_003",
    "Organize_The_Seasoning_Shelf_20250620_001",
    "Organize_The_Seasoning_Shelf_20250717_006",
    "Organize_The_Signs20250618_001",
    "Organize_The_Toiletries_20250719_008",
    "Organize_The_Wall-mounted_Coat_Rack_20250716_006",
    "Organize_Toys_20250628_002",
    "Organize_Toys_20250716_006",
    "Organize_trays_20250621_001",
    "Organizing_Desktop_Trash_20250630_003",
    "Pack_Medicines_Into_Medicine_Box_20250707_005",
    "Pen_Case_For_Storing_Stationery_20250717_006",
    "Pen_Case_For_Storing_Stationery_20250719_008",
    "Pen_Case_For_Storing_Stationery_20250802_012",
    "Pick_And_Place_Items_On_The_Potato_Chip_Shelves_20250701_003",
    "Pick_Up_Garbage_On_The_Ground_20250724_010",
    "Pick_Up_Garbage_On_The_Ground_20250801_012",
    "Pick_Up_Garbage_On_The_Ground_20250805_011",
    "Pick_Up_Or_Put_Down_The_Clip20250619",
    "Pick_Up_Trash_From_Ground_20250709_006",
    "Place_Slippers_In_The_Layered_Shoe_Cabinet_20250805_012",
    "Place_The_Dress_Shirt_On_The_Hanger20250619_001",
    "Plug_And_Unplug_Switch_Lights_20250726_010",
    "Plug_And_Unplug_Switch_Lights_20250802_012",
    "Plug_Into_A_Fixed_Socket_20250722_008",
    "Plug_The_Socket_20250616_001",
    "PourWater_20250614",
    "Pour_Leftovers_20250802_012",
    "Pour_Out_The_Leftover_Food_20250714_006",
    "Pour_The_Leftover_Food_20250724_010",
    "Pour_Water20250708_006",
    "Pour_Water_20250801_012",
    "Pour_Water_20250806_011",
    "Pull_Out_The_Cap_20250725_010",
    "Push_And_Pull_The_Chair_20250728_010",
    "Push_In_Chairs_20250614_001",
    "Pushing_Garbage_Trucks_To_Put_Garbage_Away20250707_005",
    "Put_A_Bag_In_The_Trash_Can20250617_001",
    "Put_Clothes_Into_The_Basket_20250709_006",
    "Put_Clothes_Into_The_Basket_20250731_012",
    "Put_Clothes_Into_The_Basket_20250813_011",
    "Put_Clothes_Into_Wardrobe_20250707_005",
    "Put_Shoes_Into_The_Shoe_Box_20250801_012",
    "Put_Slippers_Into_Floor_Standing_Shoe_Cabinet_20250620_001",
    "Put_The_Flowers_In_The_Vase20250618_001",
    "Put_The_Glasses_In_The_Glasses_Case_20250722_008",
    "Put_The_Glasses_In_The_Glasses_Case_20250731_012",
    "Put_The_Landline_Phone_Back_In_Place_20250725_010",
    "Put_The_Pen_Into_The_Pen_Holder_20250722_008",
    "Put_The_Pen_Into_The_Pen_Holder_20250724_010",
    "Put_The_Pen_Into_The_Pen_Holder_20250731_012",
    "Put_The_Phone_Back_In_place_20250716_006",
    "Put_The_Pillow_On_The_Bed20250617_001",
    "Put_The_Shoes_Into_The_Shoe_Box_202507116_006",
    "Put_The_Shoes_Into_The_Shoebox_20250725_010",
    "Put_The_Trash_Bag_At_The_Door20250626_002",
    "Put_the_shoes_into_the_shoe_box_20250621_001",
    "ReplaceFacialTissue_20250613_001",
    "Replace_The_Toilet_Paper_Roll_20250712_006",
    "Replace_The_Toilet_Paper_Roll_20250805_012",
    "Replace_The_Toiletries_20250616_001",
    "Replace_Toilet_Paper_20250623_002",
    "Retrieving_And_Placing_Items_Inside_The_TV_Cabinet_20250726_010",
    "Retrieving_And_Placing_Items_Inside_The_TV_Cabinet_20250805_012",
    "ReturnClothesToBasket_20250613_001",
    "Ring_The_Landline20250627_002",
    "Serve_Soup_In_The_Rice_Pot_20250717_006",
    "Serve_Soup_With_A_Rice_Pot_20250801_012",
    "Sliding_Bar_Stool_20250625_002",
    "Sort_Out_Desktop_Garbage20250802_012",
    "Steam_Rice_20250711_006",
    "Steam_Rice_20250721_008",
    "Steam_Rice_20250729_010",
    "Steam_Rice_20250730_012",
    "Stir-Fry_Dishes_20250726_010",
    "Stir-fry_20250722_008",
    "Stir-frying_20250627_002",
    "Storage_Tools_20250722_008",
    "Storage_Tools_20250725_010",
    "Storage_Tools_20250802_012",
    "Storage_Towels_20250715_006",
    "Storage_Towels_20250721_008",
    "Store_Dolls_20250812_011",
    "Store_Fruit_20250811_011",
    "Store_Tableware_20250625_002",
    "Store_The_Dolls_20250719_008",
    "Switch_The_Fan_On_And_Off_20250725_010",
    "Switch_The_Fan_On_And_Off_20250804_012",
    "Switch_The_Rotating_Nightstand_And_Place_Items_20250804_012",
    "Switch_The_Sign_20250616_001",
    "Take_A_Drink_Out_Of_Fridge_And_Put_It_On_Table_20250712_006",
    "Take_And_Place_The_Portable_Power_Bank20250628_002",
    "Take_And_Put_Items_In_And_Out_Of_The_Microwave_Oven_20250731_012",
    "Take_Drinks_From_The_Refrigerator_And_Place_Them_On_The_Table_20250728_010",
    "Take_Out_The_Laundry_From_The_Washing_Machine20250618_001",
    "Take_a_bowl_from_the_cabinet_20250620_001",
    "Take_and_place_tableware_20250620_001",
    "Taking_And_Placing_Shoes_In_The_Shoe_Cabinet20250624_002",
    "Tidy_Up_Bathroom_Toiletries_20250623_002",
    "Toast_Bread_20250618_001",
    "Turn_0n_The_Food_Pot20250618_001",
    "Turn_On_And_Off_Central_Air_Conditioning_System_20250623_002",
    "Turn_On_And_Off_Central_Air_Conditioning_System_20250624_002_special",
    "Turn_On_And_Off_The_Small_Fan_20250711_006",
    "Turn_On_And_Turn_Off_The_Air_Conditioner_20250724_010",
    "Turn_On_Off_The_Light_20250619_001",
    "Twist_The_Bottle_Cap20250619_001",
    "Twist_The_Bottle_Cap_20250725_010",
    "Twist_The_Bottle_Cap_20250731_012",
    "Uncap_The_Pen_20250804_012",
    "Use_Of_Objects_20250704_003",
    "Use_The_Drawer_To_Store_The_Hair_Dryer20250625_002",
    "Washing_Machine_Washes_Clothes_20250728_010",
    "Water_Dispenser_Connected_To_Water20250707_005",
    "Wearing_A_Strip_Rack_Hanging_Clothes_For_Retrieval_20250724_010",
    "Wipe_The_Desktop_20250801_012",
    "Wipe_The_Desktop_20250804_011",
    "Wipe_The_Sewage_Stains_With_A_Ground_Cloth_20250709_006",
    "Wipe_The_Sewage_Stains_With_A_Ground_Cloth_20250725_010",
    "Wipe_The_Sewage_Stains_With_A_Ground_Cloth_20250801_012",
)

DATASET_NAMED_MIXTURES = {
    "galaxea_open_world_stage1": [
        (f"{name}/stage1/dataset", 1.0, _ROBOT_TYPE) for name in _SUBSETS
    ],
}

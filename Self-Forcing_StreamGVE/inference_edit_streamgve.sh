# change color
python inference_edit_streamgve.py \
  --data_path '../examples/A_car.mp4' \
  --save_path './outputs/A_car-yellow.mp4' \
  --src_prompt "A grey Mini Cooper is smoothly navigating a roundabout in a bustling urban area, surrounded by historic buildings and other vehicles. The camera remains stationary, capturing the car's movement through the roundabout." \
  --trg_prompt "A yellow Mini Cooper is smoothly navigating a roundabout in a bustling urban area, surrounded by historic buildings and other vehicles. The camera remains stationary, capturing the car's movement through the roundabout." \
  --src_word "A grey Mini Cooper" \
  --trg_word "A yellow Mini Cooper" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --step 15

# remove the toy
# For removal, we suggest setting `src_word` as the object to be removed (e.g., "a toy"),
# while setting `trd_word` as the background after removing the object (e.g., "cozy living room").
# `assets/edit6_FiVE_better_remove.json` provides more removal prompt examples which is modified from FiVE-Bench. 
python inference_edit_streamgve.py \
  --data_path '../examples/A_cat.mp4' \
  --save_path './outputs/A_cat-remove_toy.mp4' \
  --src_prompt "A cat is pouncing playfully on a toy in a cozy living room, with a sofa and a window in the background. The camera remains fixed, focusing on the cat's playful behavior." \
  --trg_prompt "A cat is pouncing playfully in a cozy living room, with a sofa and a window in the background. The camera remains fixed, focusing on the cat's playful behavior." \
  --src_word "a toy" \
  --trg_word "cozy living room" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --step 15

# add sunglasses
# For addition, we suggest setting `src_word` to null prompt (i.e., "") or the desired add location (e.g., "A dog"), 
# the finer the better (e.g., `src_prompt`: "A dog with a furry head is wagging ...", `src_word`: "furry head").
python inference_edit_streamgve.py \
  --data_path '../examples/A_dog.mp4' \
  --save_path './outputs/A_dog-add_sunglasses.mp4' \
  --src_prompt "A dog is wagging its tail excitedly while sitting on a sandy beach with waves crashing in the background. The camera remains fixed, focusing on the dog's joyful expression." \
  --trg_prompt "A dog wearing sunglasses is wagging its tail excitedly while sitting on a sandy beach with waves crashing in the background. The camera remains fixed, focusing on the dog's joyful expression." \
  --src_word "A dog" \
  --trg_word "wearing sunglasses" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --step 15

# first frame condition, to porcelain
python inference_edit_streamgve.py \
  --data_path '../examples/A_woman.mp4' \
  --save_path './outputs/A_woman-porcelain.mp4' \
  --src_prompt "A woman in a black dress is walking along a paved path in a lush green park, with trees and a wooden bench in the background. The camera remains fixed, capturing her steady movement." \
  --trg_prompt "A porcelain woman is walking along a paved path in a lush green park, with trees and a wooden bench in the background. The camera remains fixed, capturing her steady movement." \
  --src_word "A woman in a black dress" \
  --trg_word "A porcelain woman" \
  --first_frame_edit ../examples/A_woman-porcelain.png \
  --fg_boost_factor 2 \
  --blend_power 2 \
  --step 15

# long video, tiger to elephant
# Self Forcing samples with a rollout strategy for long videos.
python inference_edit_streamgve.py \
  --data_path '../examples/long_tiger.mp4' \
  --save_path './outputs/long_tiger-elephant.mp4' \
  --src_prompt "At daytime, a white tiger walks among dense green trees on forest ground, seen at eye level from a static camera; neutral daylight filters through foliage and casts soft, dappled light, the background remains unobtrusive with trunks and leaves, and no additional animals or objects draw attention." \
  --trg_prompt "At daytime, an elephant walks among dense green trees on forest ground, seen at eye level from a static camera; neutral daylight filters through foliage and casts soft, dappled light, the background remains unobtrusive with trunks and leaves, and no additional animals or objects draw attention." \
  --src_word "white tiger" \
  --trg_word "elephant" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --step 5

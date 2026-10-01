"""Classify a sentence. Write your own version in your folder.

Outline (do it however you like):
- Load your saved model and label map from runs/.
- Take a sentence from the command line, print JSON with the text,
  selected_intent and confidence.
- Low confidence -> answer "unhandled" instead of guessing.
- For the five name intents (Get_coordinates, Set_navigation_target, Add_waypoint,
  Delete_waypoint, Complete_task) also pull out the name as "slot".
"""

valid_responses = ["yes", "y", "no", "n"]
negative_responses = ["no", "n"]
positive_responses = ["yes", "y"]

invalid_response_id = -1
negative_response_id = 0
positive_response_id = 1
satisfied_response_id = 3
quit_response_id = 4
retry_response_id = 5
forced_quit_response_id = 6

def process_user_input(user_input):
    user_input = user_input.lower()
    if user_input in valid_responses:
        if user_input in positive_responses:
            return positive_response_id
        elif user_input in negative_responses:
            return negative_response_id
    else:
        return invalid_response_id

def response_within_range(numeric_response, numeric_range):
    if (numeric_response >= numeric_range[0]) and (numeric_response <= numeric_range[1]):
        return True
    else:
        return False

def get_numeric_input_in_range(prompt, numeric_range):
    usr_response = invalid_response_id
    while(usr_response == invalid_response_id):
        usr_response = float(input(prompt))
        if response_within_range(usr_response, numeric_range):
            return usr_response
        else:
            print("Invalid input try again.")
            usr_response = invalid_response_id
        

def ask_if_user_has_replacement_for_requirement(requirement):
    prompt = f"This process requires {requirement} which does not currently"
    prompt += f" exist. If you dont have a replacement the program will"
    prompt += f" be forced to exit. Do you have a replacement? "
    response_id = ask_user_for_yes_or_no(prompt)
    return response_id

def ask_user_to_try_again_or_quit():
    prompt = "Would you like to try this step again?\n"
    prompt += "If you say no the program will assume you are done and exit. "
    response_id = ask_user_for_yes_or_no(prompt)
    if response_id == positive_response_id:
        return retry_response_id
    elif response_id == negative_response_id:
        return quit_response_id

def ask_user_for_yes_or_no(prompt):
    response_id = invalid_response_id
    while(response_id == invalid_response_id):
        user_input = input(prompt)
        response_id = process_user_input(user_input)
        if response_id == invalid_response_id:
            print("Invalid response try again.")
            print("We expect either yes or no.")
        else:
            return response_id

def check_if_user_satisified():
    prompt = "Are you satisfied with the displayed output"
    prompt += " for this step? "
    response_id = ask_user_for_yes_or_no(prompt)
    if response_id == positive_response_id:
        return True
    elif response_id == negative_response_id:
        return False

def get_user_feedback_for_node_output():
    user_is_satisfied = check_if_user_satisified()
    if user_is_satisfied:
        return satisfied_response_id
    else:
        response_id = ask_user_to_try_again_or_quit()
        return response_id

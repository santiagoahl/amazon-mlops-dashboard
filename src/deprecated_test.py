def convert_string(input_string: str, start_position: int, end_position: int) -> str:   
    """
    Converts string digits into capitalized word representation within a range of characters.

    Args:
        input_string (str): String to be converted.
        start_position (int): Start position to convert the string.
        end_position (int): End position to convert the string.
    
    Returns:
        str: Convert string.
    
    Example:
        >>> convert_string(string="c0nv3rt m3", start_position=1, end_position=4)
        'cZEROnv3rt m3'
    """
    
    # Verify if the start and end position are allowed
    if start_position < 1 or end_position > len(input_string):
            raise ValueError("start_position or end position values not allowed")
     
    # Select the slice of string before the start position   
    new_string = input_string[:start_position]
    digit_map = {
       "0": "ZERO", 
       "1": "ONE", 
       "2": "TWO", 
       "3": "THREE", 
       "4": "FOUR", 
       "5": "FIVE", 
       "6": "SIX", 
       "7": "SEVEN", 
       "8": "EIGTH", 
       "9": "NINE", 
    }
    
    # Map the characters within the desired range of characters 
    # and append to the new string
    for index in range(start_position, end_position):
        temp_character = input_string[index]
        
        if temp_character.isdigit():
            mapped = digit_map[temp_character]
            new_string += mapped
        else:
            new_string += temp_character
      
    # Finally add the slice of string after the end position   
      
    new_string +=  input_string[end_position:]
    return new_string

def main() -> None:
    string = input("Type your string: ")
    start_position = int(input("Type the start position (integer): "))
    end_position = int(input("Type the end position (integer): "))
    new_string = convert_string(string, start_position, end_position)
    print("Your transformed string is {}".format(new_string))

if __name__=="__main__":
    main()
/**
 * @file master_thesis_fw.ino
 * @author Peter Tumfart
 * @date 4 Apr 2022
 * @brief File containing the setup and infinity loop and calls all the necessary functions and classes.
 *
 * This program should contain a simple serial handler that allows the MATLAB paradigm to toggle the LEDs
 * and read the status of the photoresistors/photodiodes.
 * @see https://github.com/ptrt/master-thesis
 */


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); // Starting the serial connection.
  delay(1000); // Necessary to avoid double printing of header.
  print_header();
}

void loop() {
  // put your main code here, to run repeatedly:
  handle_serial();
  
}



/**
  @brief Handles the extracted command and calls the corresponding functions.
*/
void command_selection(char parameter, uint8_t value) {
  switch (parameter) {
    case 'h':
      print_header();
      break;

    case 'a':
      turn_on_led(value);
      break;

    case 'b':
      turn_off_led(value);
      break;

    case 'c':
      read_photodiode(value);
      break;
  }
}


void turn_on_led(char pos){
  if (pos == 'l' || pos == 'r' || pos == 'c' || pos == 't' || pos == 'b'){
    Serial.print("Turning on LED on position: ");
    Serial.println(pos);
  }
  else{
    Serial.println("Position not available!");
  }
}

void turn_off_led(char pos){
  if (pos == 'l' || pos == 'r' || pos == 'c' || pos == 't' || pos == 'b'){
    Serial.print("Turning off LED on position: ");
    Serial.println(pos);
  }
  else{
    Serial.println("Position not available!");
  }
}


void read_photodiode(char pos){
  if (pos == 'l' || pos == 'r' || pos == 'c' || pos == 't' || pos == 'b'){
    Serial.print("Reading photodiode on position: ");
    Serial.println(pos);
  }
  else{
    Serial.println("Position not available!");
  }
}



void print_header(){
  Serial.println("Printing header:");
  Serial.println("h\t\t Print header");
  Serial.println("a <l,r,c,t,b>\t Turn on LED position");
  Serial.println("b <1,r,c,t,b>\t Turn off LED position");
  Serial.println("c <l,r,c,t,b>\t Read photresistor position");
}

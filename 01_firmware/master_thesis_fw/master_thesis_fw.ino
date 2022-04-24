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

 #include "defines.h"

uint8_t led_pin[5] = {2,3,4,5,6};
uint8_t ldr_pin[5] = {A0, A1, A2, A3, A4};
char letter_map[5] = {'l', 'r', 'c', 't', 'b'};
uint16_t ldr_val[5];
bool first_time_dark[5] = {false, false, false, false, false};
bool first_time_bright[5]= {false, false, false, false, false};

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); // Starting the serial connection.
  delay(1000); // Necessary to avoid double printing of header.

  // Setup LEDs:
  for (uint8_t i=0; i<N_LED; i++){
    pinMode(led_pin[i], OUTPUT);
    digitalWrite(led_pin[i], LOW);
  }

  // Setup LDRs:
  for (uint8_t i=0; i<N_LDR; i++){
    pinMode(ldr_pin[i], INPUT);
    ldr_val[i] = analogRead(ldr_pin[i]);
  }

  digitalWrite(led_pin[0], LOW);
  
  print_header();
}

void loop() {
  // put your main code here, to run repeatedly:
  handle_serial();

  // Get all adc values and perform check if it's 
  // the first time to trigger the threshold:
  for (uint8_t i=0; i<N_LDR; i++){
    ldr_val[i] = analogRead(ldr_pin[i]);
    check_ldr(ldr_val[i], i);
  }  
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
    digitalWrite(get_array_index(pos), HIGH);
    Serial.print("Turning on LED on position: ");
    Serial.print(pos);
    Serial.print("\r");
  }
  else{
    Serial.print("Position not available!\r");
  }
}

void turn_off_led(char pos){
  if (pos == 'l' || pos == 'r' || pos == 'c' || pos == 't' || pos == 'b'){
    digitalWrite(get_array_index(pos), LOW);
    Serial.print("Turning off LED on position: ");
    Serial.print(pos);
    Serial.print("\r");
  }
  else{
    Serial.print("Position not available!\r");
  }
}


void read_photodiode(char pos){
  if (pos == 'l' || pos == 'r' || pos == 'c' || pos == 't' || pos == 'b'){
    Serial.print("Reading photodiode on position: ");
    Serial.print(pos);
    Serial.print(" val: ");
    Serial.print(ldr_val[get_array_index(pos)]);
    Serial.print("\r");
  }
  else{
    Serial.print("Position not available!\r");
  }
}


uint8_t get_array_index(char letter){
  for (uint8_t i=0; i<N_LED; i++){
    if (letter == letter_map[i]){
      return i;
    }
  }
}

void check_ldr(uint16_t current, uint8_t index){
  if (!first_time_dark[index] && current < THRESHOLD){
    // Signalise finster:
    first_time_dark[index] = true;
    first_time_bright[index] = false;
    Serial.print(letter_map[index]); Serial.print(" 0\r");
  }

  if (!first_time_bright[index] && current > THRESHOLD){
    // Signalise hell:
    first_time_bright[index] = true;
    first_time_dark[index] = false;
    Serial.print(letter_map[index]); Serial.print(" 1\r");
  }
}

void print_header(){
  Serial.print("Printing header:\r");
  Serial.print("h\t\t Print header\r");
  Serial.print("a <l,r,c,t,b>\t Turn on LED position\r");
  Serial.print("b <1,r,c,t,b>\t Turn off LED position\r");
  Serial.print("c <l,r,c,t,b>\t Read photresistor position\r");
}

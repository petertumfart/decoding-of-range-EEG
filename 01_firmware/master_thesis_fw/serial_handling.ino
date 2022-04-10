 /**
 * @file serial_handling.ino
 * @brief Holds all the function that are necessary to interact with the
 *        PC-SW (Matlab).
 */

#include "defines.h"


 
/**
  @brief Handles incomming serial data
*/
void handle_serial( ){
  char command[ARRAY_SIZE] = {};
  uint8_t i = 0;
  bool correct_command = false;

  // Read everything that is available in the buffer:
  if (Serial.available() > 0){
    while (Serial.available() > 0){
      command[i] = Serial.read();
      if (command[i] == '\r') {
        break;
      }
      delay(3);
      i++;
    }
    command_selection(command[0], command[2]);
  }
}

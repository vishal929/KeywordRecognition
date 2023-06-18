#include <bluefruit.h>
#include <Servo.h>
#include <Adafruit_TinyUSB.h> // for Serial

// BLE services needed
BLEUart bleuart; // uart client (receiving information from raspberry pi)
BLEBas blebas; // ble battery service
BLEDis bledis; // device info
BLEDfu bledfu; // OTA DFU service


// servos
Servo theater;
Servo bar;

// trigger bytes for switching the servos
const char* theater_name = "theater";
const char* bar_name = "bar";
// our read buffer (raw data)
char raw_buffer[64];
// our name buffer (this is formatted to remove whitespace,etc.)
char name_buffer[64];
// number of bytes available in blueart in event loop
int num_bytes=0;
// number of bytes read
int num_read=0;


// flags for servo status (on or off basically to flip a switch)
bool theater_on = false;
bool bar_on = false;



void setup() {
  theater.attach(12);
  bar.attach(13);
  // put your setup code here, to run once:
  Serial.begin(115200);

  // we do not want the led on for power savings
  Bluefruit.autoConnLed(true);

 

  // set max bandwidth for the connection
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);

  Bluefruit.begin();

  
  Bluefruit.setTxPower(8);
  Bluefruit.setName("ItsyBitsy Theater+Bar");
  Bluefruit.Periph.setConnectCallback(connect_callback);
  Bluefruit.Periph.setDisconnectCallback(disconnect_callback);

  bledfu.begin();

  bledis.setManufacturer("Adafruit Industries");
  bledis.setModel("Bluefruit ItsyBitsy NRF52840 Feather");
  bledis.begin();

  bleuart.begin();

  blebas.begin();
  blebas.write(100);

  startAdv();

  
}

void startAdv(void)
{
  // Advertising packet
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();

  // Include bleuart 128-bit uuid
  Bluefruit.Advertising.addService(bleuart);

  // Secondary Scan Response packet (optional)
  // Since there is no room for 'Name' in Advertising packet
  Bluefruit.ScanResponse.addName();

  
  /* Start Advertising
   * - Enable auto advertising if disconnected
   * - Interval:  fast mode = 20 ms, slow mode = 152.5 ms
   * - Timeout for fast mode is 30 seconds
   * - Start(timeout) with timeout = 0 will advertise forever (until connected)
   * 
   * For recommended advertising interval
   * https://developer.apple.com/library/content/qa/qa1931/_index.html   
   */
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);    // in unit of 0.625 ms
  Bluefruit.Advertising.setFastTimeout(30);      // number of seconds in fast mode
  Bluefruit.Advertising.start(0);                // 0 = Don't stop advertising after n seconds  
}

void loop() {
  // put your main code here, to run repeatedly:
  
  // just need to listen for data, no need to send anything
  while (bleuart.available()){
    num_bytes = bleuart.available();
    Serial.println(num_bytes);
    // error handling
    if (num_bytes + num_read > sizeof(raw_buffer)){
      // then we have something invalid in the bleuart buffer
      bleuart.flush();
      continue;
    }

    // reading into our buffer
    num_read += bleuart.read(raw_buffer,num_bytes);
    
  }

  // need to explicitly set \0 (null terminator) for some reason
  raw_buffer[num_read]='\0';

  // resetting num_read
  num_read=0;
  bleuart.flush();

  if (strlen(raw_buffer)>0){
    // putting formatted data into the name_buffer
    formatstr(raw_buffer, name_buffer);
    Serial.printf("%s----%s \n", raw_buffer,name_buffer);
    if (strcmp(name_buffer,theater_name)==0){
      // switch the theater servo
      Serial.println(" flipped theater!");
      theater_on = !theater_on;
      flip_switch(&theater,theater_on);
    }
    if (strcmp(name_buffer,bar_name)==0){
      // switch the bar servo
      Serial.println(" flipped bar!");
      bar_on = !bar_on;
      flip_switch(&bar, bar_on);
    }  
    // empty our buffers
    name_buffer[0] = '\0';
    raw_buffer[0] = '\0';
    
  }
  
  // small delay just for power optimization
  delay(100);
}

// callback invoked when central connects
void connect_callback(uint16_t conn_handle)
{
  // Get the reference to current connection
  BLEConnection* connection = Bluefruit.Connection(conn_handle);

  char central_name[32] = { 0 };
  connection->getPeerName(central_name, sizeof(central_name));
}

/**
 * Callback invoked when a connection is dropped
 * @param conn_handle connection where this event happens
 * @param reason is a BLE_HCI_STATUS_CODE which can be found in ble_hci.h
 */
void disconnect_callback(uint16_t conn_handle, uint8_t reason)
{
  (void) conn_handle;
  (void) reason;
}


/*
  Logic to flip a switch with a servo
  the boolean on/off represents the position of the servo,
  which corresponds to flipping a physical switch
*/
void flip_switch(Servo *servo, bool on){
  if (on){
    servo->write(30);
  } else{
    servo->write(170);
  }
}

// function to format a string how we want
// 1) remove whitespace and non alphabet characters
// 2) convert alphabet characters to whitespace
void formatstr(char* str, char* out_buffer){
  int num_formatted = 0;
  for (int i=0;;i++){
    if (str[i]=='\0'){
      out_buffer[num_formatted] = '\0';
      return;
    }
    if (isalpha(str[i])){
      out_buffer[num_formatted++] = tolower(str[i]);
    }
  }
}

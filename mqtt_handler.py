import paho.mqtt.client as mqtt

def setup_mqtt(broker, port):
    """
    Set up and connect an MQTT client.
    """
    client = mqtt.Client()
    client.connect(broker, port)
    return client

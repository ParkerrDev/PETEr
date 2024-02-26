# from encoder import encoder


class winch:
    def __init__(self):
        super().__init__()

    def get_position():
        # with open("cache/winch_position.json):
        #           winch_position = value
    def move_up(distance):
        winch_position = self.get_position()
        encoder.setValue(winch_position)
        destination = winch_position + distance
        while encoder.position != destination:
            #move winch ccw
    def move_down(distance):
        winch_position = self.get_position()
        encoder.setValue(winch_position)
        destination = winch_position - distance
        while encoder.position != destination:
            #move winch cw
    def go_to(encoder_position):
        winch_position = self.get_position()
        encoder.setValue(winch_position)
        # fancy logic to determine if going up or down and by how much
    def go_to_zero():
    def extend_max():

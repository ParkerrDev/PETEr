from subsystems import battery, winch, pathfinding, spectrophotometer
import asyncio


async def run():
    
    battery_monitor.start()
    winch.go_to_zero()
    winch.extend_max()
    pathfinding.start()
    spectro_analyisis.start()
    # filtration.start()
    web_display.start()


if __name__ == "__main__":
    async run()

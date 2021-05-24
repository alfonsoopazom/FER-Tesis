import sc2
from sc2 import Race, run_game, maps, Difficulty
from sc2.player import Bot, Computer


class agente(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()

    def location(self):
        locx = self.start_location.x
        locy = self.start_location.x
        print(locx)
        return locy, locx


run_game(maps.get('MoveToBeacon'), [
    Bot(Race.Protoss, agente()),
], realtime=True)

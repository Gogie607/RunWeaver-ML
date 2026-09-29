class MetricHandlerBase:


    def begin(self, params=None):
        self.reset()

    def reset(self):
        #print(f"\nResetting metric handlers...")
        pass

    def update(self, payload):
        #print(f"\nUpdating metric handlers...")
        pass
    def summarize(self) -> dict:
        #print(f"\nSummarizing metric handlers...")
        return {}

    def report(self, s=None) -> str:
        return ""
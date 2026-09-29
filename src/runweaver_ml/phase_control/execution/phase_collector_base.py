class CollectorBase:

    def begin(self, params=None, step=None):
        pass

    def update(self, payload):
        pass

    def finalize(self):
        return None

    def save(self):
        return None
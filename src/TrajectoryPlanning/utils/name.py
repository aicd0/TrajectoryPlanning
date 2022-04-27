class Name:
    def __init__(self, prefix: str) -> None:
        self.cls_num = {}
        self.prefix = prefix
        if len(self.prefix) > 0:
            self.prefix += '_'

    def get(self, cls_name: str, name: str | None) -> str:
        if name is None:
            if cls_name in self.cls_num:
                self.cls_num[cls_name] += 1
            else:
                self.cls_num[cls_name] = 1
            ans = self.prefix + cls_name + '_' + str(self.cls_num[cls_name])
        else:
            ans = self.prefix + name
        assert len(ans) > 0
        return ans
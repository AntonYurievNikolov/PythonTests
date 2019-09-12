import service


def main():
    print("Welcome to the talk python info downloader")
    print()
    service.download_info()
    for show_id in range(100, 130):
        info = service.get_episode(show_id)
        print(f"{info.show_id}. {info.title}")


if __name__ == '__main__':
    main()


class CreatureBase:
    def __init__(self, name, level):
        self.name = name
        self.level = level

    def __repr__(self):
        return f"CreatureBase: {self.name} level: {self.level}"


class Wizard(CreatureBase):
    pass


class Dragon(CreatureBase):
    pass

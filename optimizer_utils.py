def show_optimizer_structure(optimizer_odict):
    """
    show optimizer names and its structure
    input: ordered dictionary containing models
    """
    for name, optimizer in optimizer_odict.items():
        print('optimizer name:', name)
        print('optimizer specification:', optimizer)
        print("")
    return


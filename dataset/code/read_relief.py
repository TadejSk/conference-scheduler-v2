f = open('relief_best.txt')
content = f.readlines()
features = [int(x) for x in content]
features = sorted(features)
for f in features:
    print(f)
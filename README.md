# ML_Module_01

## for executing file
```
python3 <filename>
```

## for checking format
```
pycodestyle <filename>
```

## for modifying format
```
black <filename>
```

## for copying docker file
```
docker cp <docker container id>:/app/results .
docker exec -it <CONTAINER_ID> ./test.sh
docker cp <CONTAINER_ID>:/app/results .
```

## tips
- @staticmethod: A decorator in Python that defines a static method in a class. It allows you to write methods inside a class that are not tied to an instance of the class. No self Parameter.
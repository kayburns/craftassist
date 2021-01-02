# Use Minecraft Render

We use [Chunky](https://github.com/chunky-dev/chunky) to render everything in minecraft world.

For License issue, we can not include Chunky code in our repo. Please download ChunkyLauncher [here](https://github.com/chunky-dev/chunky) and put it under python/minecraft_render in order to use minecraft renderer.

Install Java 8.
```
apt-get install openjdk-8-jdk
# verify
java -version
# if you get an error about javafx, install Java 8 compatible version
apt purge openjfx
apt install openjfx=8u161-b12-1ubuntu2 libopenjfx-jni=8u161-b12-1ubuntu2 libopenjfx-java=8u161-b12-1ubuntu2
apt-mark hold openjfx libopenjfx-jni libopenjfx-java
# verify
find / -name 'javafx.properties'
# inspect files from previous command, e.g.:
cat $JAVA_HOME/jre/lib/javafx.properties
```

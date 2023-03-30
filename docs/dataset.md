# Download Objaverse
Use the [provided script](../download_objaverse.py) to download [objaverse](https://objaverse.allenai.org/).

```bash
    cd /Root/of/the/repository
    python download_objaverse.py --root [/path/to/objaverse/root]
```
In the process, the annotation and .glb files will be cached in $root/.objaverse, you could modify the BASE_PATH [here](../objaverse/__init__.py#L12) to change the cached directory to wherever you want.

With the script, all objects with assigned LVIS categories will be downloaded into the root, structured as follows:

## Dataset Structure
The dataset should be structured as follows:
```bash
  DATA_ROOT(For Objaverse)
  ├── data
  |     ├── category1
  |     |       ├── uid0
  |     |       |   ├── image.jpeg
  |     |       |   |── 3Dobject.glb
  |     |       ├── uid1
  |     |       .
  |     |       .
  |     ├── category2
  |     |       ├── uid6
  |     |       ├── uid7  
  ├── annotations.json
```

## Annotation Format
See [here](https://objaverse.allenai.org/docs/download#loading-annotations) for more details.
The annotation of each object could be accessed with uid.
The annotation is formatted as follows:

### Attributes:
0. uri
1. uid
2. name
3. staffpickedAt
4. viewCount
5. likeCount
6. animationCount
7. viewerUrl
8. embedUrl
9. commentCount
10. isDownloadable
11. publishedAt
12. tags
13. categories
14. thumbnails
15. user
16. description
17. faceCount
18. createdAt
19. vertexCount
20. isAgeRestricted
21. archives
22. license

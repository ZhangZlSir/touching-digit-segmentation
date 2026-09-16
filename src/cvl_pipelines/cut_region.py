#cut_region.py
import numpy as np

#本函数使用了集合，不适合@jit加速
def find_connected_points(image, x_range=(13, 27)):
    height, width = image.shape

    found_points = set()
    
    min_val = float('inf')
    min_x = None
    for x in range(x_range[0], x_range[1] + 1):
        if x < width:
            val = image[0, x]
            if val < min_val:
                min_val = val
                min_x = x
    
    if min_x is not None:
        found_points.add((min_x, 0))
    
    def generate_candidates(points_set):
        candidates = set()
        for (x, y) in points_set:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_x = x + dx
                    new_y = y + dy
                    if (new_y >= 0 and new_y < height and 
                        x_range[0] <= new_x <= x_range[1] and 
                        new_x < width):
                        candidates.add((new_x, new_y))
        return candidates
    
    prev_points = set(found_points)
    
    while True:
        candidates = generate_candidates(found_points)
        candidates = {(x, y) for (x, y) in candidates if (x, y) not in found_points}
        
        if not candidates:
            break
        
        zero_candidates = [ (x, y) for (x, y) in candidates if image[y, x] == 0 ]
        
        if zero_candidates:
            new_points = set(zero_candidates)
        else:
            min_val = float('inf')
            min_candidates = []
            for (x, y) in candidates:
                val = image[y, x]
                if val < min_val:
                    min_val = val
                    min_candidates = [(x, y)]
                elif val == min_val:
                    min_candidates.append((x, y))
            
            if not min_candidates:
                break
            new_points = set(min_candidates)
        
        if not new_points:
            break
        
        found_points.update(new_points)
        
        if any(y == height - 1 for (x, y) in new_points):
            break
    
    points = sorted(list(found_points), key=lambda p: p[1])
    return points
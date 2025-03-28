

def add_vectors(vector1, vector2):
    return {"x": vector1["x"] + vector2["x"], "y": vector1["y"] + vector2["y"]}

def decrease_vectors(vector1, vector2):
    return {"x": vector1["x"] - vector2["x"], "y": vector1["y"] - vector2["y"]}

def trajectory_list_to_vector_list(trajectory_list):
    """
    Convert a list of trajectories (each a list of coordinates) to a list of trajectories of movement vectors.
    For each trajectory, compute the vector from coordinate[i] to coordinate[i+1] (and assign {0, 0} for the last frame).
    """
    subject_vectors_list = []
    for traj in trajectory_list:
        vectors = []
        num_frames = len(traj)
        for i in range(num_frames - 1):
            dx = traj[i+1]["x"] - traj[i]["x"]
            dy = traj[i+1]["y"] - traj[i]["y"]
            vectors.append({"x": dx, "y": dy})
        # For the last frame, there is no next coordinate so use a zero vector.
        vectors.append({"x": 0, "y": 0})
        subject_vectors_list.append(vectors)
    return subject_vectors_list
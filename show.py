import snake_realist as snake
import pickle

def show():
    print("Showing save...")
    with open("./showoff.pkl",'rb') as f:
        m = pickle.load(f)

    env = snake.board()
    k = input("Show output")
    env.reset(fpos=m[1])

    env.render(m[0],m[1])
    k=input("Done!")
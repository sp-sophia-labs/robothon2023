def main():

    def revise_local_max(local_max):
          
        revised_local_max = []
        base_x = -1
        for index, x in enumerate(local_max):
            if base_x < 0:
                if index == len(local_max) - 1:
                    revised_local_max.append(x)
                    break
                if x != local_max[index + 1] -1:
                    revised_local_max.append(x)
                else:
                    base_x = x
            else:
                if index == len(local_max) - 1:
                    revised_local_max.append((base_x + x) / 2)
                    break
                if x != local_max[index + 1] - 1:
                    revised_local_max.append((base_x + x) / 2)
                    base_x = -1
                    
        return revised_local_max
    
    local_max_1 = [39, 40, 41, 50, 56]
    local_max_2 = [30, 40, 41, 54]
    local_max_3 = [30, 40, 50]
    local_max_4 = [23, 34, 35, 37]
    
    print(local_max_1)
    print(revise_local_max(local_max_1))
    print('-------------')
    print(local_max_2)
    print(revise_local_max(local_max_2))
    print('-------------')
    print(local_max_3)
    print(revise_local_max(local_max_3))
    print('-------------')
    print(local_max_4)
    print(revise_local_max(local_max_4))
    print('-------------')

if __name__ == "__main__":

    main()
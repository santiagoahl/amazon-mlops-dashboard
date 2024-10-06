if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(
    """{sum_a_b}\n{diff_a_b}\n{prod_a_b}""".format(
        sum_a_b=a+b,
        diff_a_b=a-b,
        prod_a_b=a*b
        )
    )
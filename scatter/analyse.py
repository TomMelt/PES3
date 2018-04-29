import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import constants as c


headers = "seed J v i epsilon b c.R0 dist rix riy riz pix piy piz Rix Riy Riz Pix Piy Piz rfx rfy rfz pfx pfy pfz Rfx Rfy Rfz Pfx Pfy Pfz R1 R2 R3 KE1i KE2i KE1f KE2f tf countstep maxstep maxErr Hi Hf countElastic countTotal countAB countAC countBC" # noqa
headers = headers.split(" ")


def analyse(infiles):

    # import data from input files
    frames = []
    runningTotal = np.zeros(5, dtype=int)
    for infile in infiles:
        df = pd.read_csv(infile, sep=" ", header=None, names=headers)
        for col in range(5):
            df.iloc[:, -(1+col)] = df.iloc[:, -(1+col)].apply(lambda x: x + runningTotal[col])
            runningTotal[col] = df.iloc[-1, -(1+col)]
        frames.append(df)

    # compile data into one dataframe
    df = pd.concat(frames)
    df = df.reset_index(drop=True)

    df['AB'] = df.apply(lambda x: x['R2'] == min(x['R1'], x['R2'], x['R3']), axis=1)

    print(df.loc[df['AB'] == True])

    epsilon = df['epsilon'].iloc[0]

    # ========== plotting ==========

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(231)
    df['countReact'] = df['countAB']+df['countAC']
    df['cross'] = np.pi*c.bmax*c.bmax*df['countReact']/df['countTotal']
    df['cross'].plot()
    df['crossErr'] = df['cross']*np.sqrt((df['countTotal']-df['countReact'])/df['countTotal']*df['countReact'])
    plt.fill_between(df.index, df['cross']-df['crossErr'], df['cross']+df['crossErr'], color='b', alpha=0.2)
    ax.set_xlabel('no. of trajs')
    ax.set_ylabel('$\sigma$ (a.u.)')

    ax = fig.add_subplot(232)
    plt.hist2d(df['b'], np.abs(df['KE2i']-df['KE2f']), bins=10)
    ax.set_xlabel('b')
    ax.set_ylabel(r'$\mathrm{abs}(E_{ki}^2-E_{kf}^2)$')
    ax.set_title('rtol='+str(c.rtol)+' atol='+str(c.atol)+' E='+str(epsilon))

    ax = fig.add_subplot(233)
    df['maxErr'].apply(lambda x: np.log10(x)).plot.hist()
    ax.set_xlabel(r'$\log_{10}(H_{error})$')
    ax.set_ylabel('count')

    ax = fig.add_subplot(234)
    df['maxstep'].plot.hist()
    ax.set_xlabel('maxstep size (a.u.)')
    ax.set_ylabel('count')

    ax = fig.add_subplot(235)
    df['tf'].plot.hist()
    ax.set_xlabel(r'$t_f$ (a.u.)')
    ax.set_ylabel('count')

    ax = fig.add_subplot(236)
    df['countstep'].plot.hist()
    ax.set_xlabel('no. of steps')
    ax.set_ylabel('count')


    plt.show()

    df.to_csv("compiled.csv")

    return


if __name__ == "__main__":
    analyse(sys.argv[1:])

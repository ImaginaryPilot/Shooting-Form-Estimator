import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

basePath = Path(".")
JOINTS = [29, 30, 31, 32]

def loadFolder(folder):
    # Paths
    correctPhasesPath = folder / "CorrectData2D.csv"
    ballDataPath = folder / "DataBall2D.csv"
    # csvPath = folder / "Data2D.csv"

    if not correctPhasesPath.exists():
        print(f"Skipping {folder}: CorrectData2D.csv does not exist.")
        return None

    # Read Correct Data, Data and Ball Data
    correctDf = pd.read_csv(correctPhasesPath, index_col=0)
    phases = (correctDf.loc["META_SEPARATE"].dropna().astype(int).tolist())

    startFrame = phases[0]
    endFrame = phases[4]

    numericFrames = pd.to_numeric(correctDf.index, errors="coerce")
    mask = (numericFrames >= startFrame) & (numericFrames <= endFrame)

    correctData = correctDf.loc[mask].copy()
    correctData.index = numericFrames[mask].astype(int)

    # df = pd.read_csv(csvPath, index_col=0)
    ballDf = pd.read_csv(ballDataPath).dropna()

    return correctData, phases, ballDf

# Calculate Ball Data
def ballCalc(df):
    df = df.sort_values("Frame")
    df = df.set_index("Frame")
    frameDiff = pd.Series(df.index, index=df.index).diff()

    diffX = df["BallX"].diff()
    diffY = df["BallY"].diff()

    velX = diffX / frameDiff
    velY = diffY / frameDiff
    vel = np.sqrt(velX**2 + velY**2)

    aclX = velX.diff() / frameDiff
    aclY = velY.diff() / frameDiff
    acl = np.sqrt(aclX**2 + aclY**2)

    ballData = pd.DataFrame({
        "BallX": df["BallX"],
        "BallY": df["BallY"],
        "FrameDiff": frameDiff,
        "DiffX": diffX,
        "DiffY": diffY,
        "VelocityX": velX,
        "VelocityY": velY,
        "Velocity": vel,
        "AccelerationX": aclX,
        "AccelerationY": aclY,
        "Acceleration": acl
    })
    
    return ballData

def jointCalc(df):
    jointDf = {}

    for joint in JOINTS:
        coordX = df[f"{joint}X"]
        coordY = df[f"{joint}Y"]
        velX = coordX.diff().shift(-1)
        velY = coordY.diff().shift(-1)
        vel = np.sqrt(velX**2 + velY**2)
        aclX = coordX.shift(-1) - 2 * coordX + coordX.shift(1)
        aclY = coordY.shift(-1) - 2 * coordY + coordY.shift(1)
        acl = np.sqrt(aclX**2 + aclY**2)
        
        jointDf[joint] = pd.DataFrame({
            "X": coordX,
            "Y": coordY,
            "Velocity": vel,
            "Acceleration": acl,
            "DotProduct": velX*aclX + velY*aclY
        })

    return jointDf

def groundDetecter(joint, threshold=15, minFrames=4):
    vel = joint["Velocity"] <= threshold
    
    groundRegion = []
    startFrame = None
    previousFrame = None
    
    for frame, isGrounded in vel.items():
        if isGrounded and startFrame is None:
            startFrame = frame

        elif not isGrounded and startFrame is not None:
            duration = previousFrame - startFrame + 1
            
            if duration >= minFrames:
                groundRegion.append((startFrame, previousFrame))
            startFrame = None

        previousFrame = frame

    if startFrame is not None:
        duration = previousFrame - startFrame + 1
        
        if duration >= minFrames:
            groundRegion.append((startFrame, previousFrame))
    
    return groundRegion

def mergeRegions(heelRegions, toeRegions):
    regions = sorted(heelRegions + toeRegions)
    merged = []
    for start, end in regions:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))

        else:
            previousStart, previousEnd = merged[-1]
            merged[-1] = (previousStart, max(previousEnd, end))

    return merged

def intervalSpacing(regions):
    spacing = []
    for i in range(len(regions) - 1):
        current_end = regions[i][1]
        next_start = regions[i+1][0]
        spacing.append(next_start - current_end)

    return spacing

def plantFootInterval(plantRegions, kickRegions):
    spacing = intervalSpacing(kickRegions)

    if not spacing:
        return None

    maxIndex = spacing.index(max(spacing))
    gapStart = kickRegions[maxIndex][1]

    for start, end in plantRegions:
        if start>= gapStart:
            return(start,end)

    return None

def plantFootEstimation(plantInterval):
    if plantInterval is None:
        return None

    return plantInterval[0] + 2

def impactFootEstimation(plantInterval, heelRegions, toeRegions):
    if plantInterval is None:
        return None

    start, end = plantInterval

    
    heelData = heelRegions.loc[start:end].add_suffix("_Heel")
    toeData = toeRegions.loc[start:end].add_suffix("_Toe")

    kickingData = pd.concat([heelData, toeData], axis=1)

    print(kickingData)


def accelerationDectector(joint, switch=True, minFrames=3):
    plantAcceleration = joint["DotProduct"]
    mask = plantAcceleration >= 0 if switch else plantAcceleration < 0

    region = []
    startFrame = None
    previousFrame = None

    for frame, isAccel in mask.items():
        if isAccel and startFrame is None:
            startFrame = frame

        elif not isAccel and startFrame is not None:
            duration = previousFrame - startFrame + 1

            if duration >= minFrames:
                region.append((startFrame, previousFrame))
            startFrame = None

        previousFrame = frame

    if startFrame is not None:
        duration = previousFrame - startFrame + 1

        if duration >= minFrames:
            region.append((startFrame, previousFrame))

    return region
    
def algA(folder, phases, jointData):
    plantX = jointData[29]["X"]
    plantSpeed = jointData[29]["Velocity"]
    shootSpeed = jointData[30]["Velocity"]

    quiet = plantSpeed.rolling(3, center=True).max() < 10
    estimatedImpactFrame = shootSpeed[quiet].idxmax()
    f = estimatedImpactFrame
    while (f - 1) in plantX.index and abs (plantX.loc[f] - plantX.loc[f - 1]) <= 6:
        f -= 1
    estimatedPlantFrame = f

    actualImpactFrame = phases[3]
    actualPlantFrame = phases[2]
    
    plantDifference = abs(estimatedPlantFrame - actualPlantFrame)
    correctPlant = plantDifference <= 1
    
    impactDifference = abs(estimatedImpactFrame - actualImpactFrame)
    correctImpact = impactDifference <= 1

    frameBefore = actualPlantFrame - 5
    frameAfter = actualPlantFrame + 5

    print(folder)
    # print("Estimated plant frame:", estimatedPlantFrame)
    print("Actual plant frame:", actualPlantFrame)
    # print("Estimated impact frame:", estimatedImpactFrame)
    print("Actual impact frame:", actualImpactFrame)
    # print(jointData[29].loc[0:frameAfter], "\n")
    # print(pd.concat({ "Joint 29": jointData[29].loc[0:phases[4]], "Joint 31": jointData[31].loc[0:phases[4]]}, axis=1))
    # print("Correct Plant:", correctPlant)
    # print("Correct Impact:", correctImpact, "\n")

    return 

# Calculate ball Velocit Spikes
def ballVelocitySpikes(df, threshold=15):
    ball = df.sort_index().copy()

    # Calculate Velocity Difference
    ball["VelocityChange"] = ball["Velocity"].diff()
    spikes = ball[(ball["Velocity"] > threshold) & (ball["VelocityChange"] > threshold)]

    if len(spikes) == 0:
        return None

    firstFrame = spikes.index[0]
    if len(spikes) > 1:
        secondFrame = spikes.index[1]
    else:
        secondFrame = None

    return {
        "First Spike" : firstFrame,
        "Second Spike": secondFrame
    }

# Find plantFootCandidates
def plantFootCandidates(jointData, ballDf, minDecelFrames=2):
    plantAcceleration = jointData[29]["Acceleration"]
    decelerating = plantAcceleration < 0

    deceleratingRegions = []
    startFrame = None
    previousFrame = None

    for frame, isDecel in decelerating.items():
        if isDecel and startFrame is None:
            startFrame = frame

        elif not isDecel and startFrame is not None:
            duration = previousFrame - startFrame + 1

            if duration >= minDecelFrames:
                deceleratingRegions.append((startFrame, previousFrame))
            startFrame = None

        previousFrame = frame

    if startFrame is not None:
        duration = previousFrame - startFrame + 1

        if duration >= minDecelFrames:
            deceleratingRegions.append((startFrame, previousFrame))

    spikes = ballVelocitySpikes(ballDf)

    # if spikes:
    #     firstSpike = spikes["First Spike"]
    #     secondSpike = spikes["Second Spike"]

    #     frameDiff = ballDf.loc[firstSpike, "FrameDiff"]

    #     movementStart = firstSpike - int(frameDiff)
    #     movementEnd = secondSpike if secondSpike is not None else firstSpike

    #     spikeInterval = (movementStart, movementEnd)

    #     overlappingRegions = [
    #         (start, end)
    #         for start, end in deceleratingRegions
    #         if start <= spikeInterval[1] and end >= spikeInterval[0]
    #     ]

    #     candidateInterval = (overlappingRegions[0][0], overlappingRegions[-1][1])
    #     print("Candidate interval:", candidateInterval)

def main():
    results = []

    players = sorted(
        player for player in basePath.iterdir()
        if player.is_dir()
        and player.name.startswith("player_")
    )

    for player in players:
        char = player.name.split("_", 1)[1]
        if char== "L":
            continue
        
        folders = sorted(
            folder for folder in player.iterdir()
            if folder.is_dir()
            and folder.name.startswith(f"{char}_")
        )

        for folder in folders:
            data = loadFolder(folder)

            if data is None:
                continue
            df, phases, ballDf = data
            jointDf = jointCalc(df)
            ballData = ballCalc(ballDf)
            pltFtHeel = jointDf[29]
            kckFtHeel = jointDf[30]
            pltFtToe = jointDf[31]
            kckFtToe = jointDf[32]

            # algA(folder, phases, jointDf)
            # plantFootCandidates(jointDf, ballData)
            pltFt = mergeRegions(groundDetecter(pltFtHeel),groundDetecter(pltFtToe))
            kckFt = mergeRegions(groundDetecter(kckFtHeel),groundDetecter(kckFtToe))
            pltFtInterval = plantFootInterval(pltFt, kckFt)
            pltFtEst = plantFootEstimation(pltFtInterval)
            print(folder)
            print("Actual Impact Foot: ", phases[3])
            # print("Heel Deceleration: ", accelerationDectector(kckFtHeel, False))
            # print("Toe Deceleration: ", accelerationDectector(kckFtToe, False))
            impactFootEstimation(pltFtInterval, kckFtHeel, kckFtToe)

            pltFtEstError = pltFtEst - phases[2]
            results.append({
                "Trial": folder.name,
                "Estimated Plant": pltFtEst,
                "Plant Interval": pltFtInterval,
                "Actual Plant": phases[2],
                "Plant Error": pltFtEstError if pltFtEst is not None else None,
                "Actual Impact": phases[3],
            })

    resultsDf = pd.DataFrame(results)
    print("\nDetection Results:")
    print(resultsDf.to_string(index=False))

    errors = resultsDf["Plant Error"].dropna().abs()
    stats = {
        "Exact": (errors == 0).sum(),
        "Off by 1": (errors == 1).sum(),
        "Off by 2+": (errors >= 2).sum(),
        "Total": len(errors)
    }

    statsDf = pd.DataFrame([stats])
    print("\nDetection Statistics:")
    print(statsDf.to_string(index=False))

    accuracy = stats["Exact"] / stats["Total"] * 100
    print(f"\nExact Detection Accuracy: {accuracy:.2f}%")
    accuracy = (stats["Exact"] + stats["Off by 1"]) / stats["Total"] * 100
    print(f"Within 1 Detection Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
open System

#I @"X:\Projects\EnergonWin8\packages\FSharp.Charting.0.90.6"
#load "FSharp.Charting.fsx"


open FSharp.Charting

// ----------- Load data --------------

let datafolder = @"X:\Projects\featureBasedScheduling\Dati"

// merge convolution files
//let csvfiles = System.IO.Directory.EnumerateFiles(datafolder, "FeatureExtractionResults.csv")
let csvfiles = System.IO.Directory.EnumerateFiles(datafolder, "FeatureExtractionResultsNoBranch.csv")
let filename = csvfiles |> Seq.head
let fieldNames =
    let lines = System.IO.File.ReadAllLines(filename)
    let header = lines |> Seq.head
    header.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)

let handleFile filename =
    let lines = System.IO.File.ReadAllLines(filename)
    let header = lines |> Seq.head
    let fields = header.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
    let handleLine (line:string) =
        let cells = line.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
        cells |> Array.map (fun f -> System.Double.Parse(f))
    lines |> Seq.skip 1 |> Seq.map handleLine

let allfields = csvfiles |> Seq.map handleFile |> Seq.concat |> Seq.toArray

fieldNames |> Array.iteri (fun i v -> System.Console.WriteLine(@"{0} {1}", i, v))

//allfields.[0].Length

let getColumn i = allfields |> Array.map (fun ary -> ary.[i])

let inline sqr x = x * x
let normColumn col:float[] = 
    let avg = col |> Array.average
    let stddev = col |> Array.map ((-) avg) |> Array.map sqr |> Array.average |> sqrt
    //col |> Array.map (fun v -> (v - avg) / stddev)
    col

let alg = getColumn 25 

let discreteGPUTime = getColumn 22
let integratedGPUTime = getColumn 23
let CPUTime = getColumn 24 

let workSize = allfields |> Array.map (fun ary -> ary.[16] * ary.[17] * ary.[18])

let dataTrasfFromHost = getColumn 13
let dataTrasfFromDevice = getColumn 14

let readAccessGlobal = getColumn 1 
let writeAccessGlobal = getColumn 2 
let readAccessLocal = getColumn 3 
let writeAccessLocal = getColumn 4 
let readAccessConstant = getColumn 5 

let mulArrays (a:float[]) (b:float[]) = Array.map2 (fun v w -> v*w) a b 

let fixWithWorkSize a = mulArrays workSize a 

let arithmetic = fixWithWorkSize (getColumn 6)
let logic = fixWithWorkSize (getColumn 7) 
let bitwise = fixWithWorkSize (getColumn 8)
let comparison = fixWithWorkSize (getColumn 9)
let pow = fixWithWorkSize (getColumn 10)
let sqrt_op = fixWithWorkSize (getColumn 11)
let hypot = fixWithWorkSize (getColumn 12)

let sumArrays (a:float[]) (b:float[]) = Array.map2 (fun v w -> v+w) a b 
let simple_op =  (sumArrays logic bitwise)  |> sumArrays comparison
let complex_op = sumArrays arithmetic pow |> sumArrays sqrt_op |> sumArrays hypot 

//let readAccessGlobal = getColumn 1
//let writeAccessGlobal = getColumn 2
//let readAccessLocal = getColumn 3
//let writeAccessLocal = getColumn 4
//let readAccessConstant = getColumn 5

let divArrays (a:float[]) (b:float[]) = Array.map2 (fun v w -> v/w) a b 

let allMem = sumArrays readAccessGlobal writeAccessGlobal |> sumArrays readAccessLocal |> sumArrays writeAccessLocal |> sumArrays readAccessConstant
let memTot = mulArrays workSize allMem
let derived = divArrays (sumArrays (dataTrasfFromHost) (dataTrasfFromDevice)) memTot

#r @"X:\Projects\featureBasedScheduling\AnalyseData\packages\MathNet.Numerics.FSharp.2.6.0\lib\net40\MathNet.Numerics.FSharp.dll"
#I @"X:\Projects\featureBasedScheduling\AnalyseData\packages\MathNet.Numerics.2.6.2\lib\net40"
#r @"MathNet.Numerics.dll"
#r @"MathNet.Numerics.IO.dll"
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra.Double

//let target = 2

//let basisSize = 2

let rnd = new System.Random()





// ------------------------------------------------------------------------------


#I @"X:\Projects\EnergonWin8\packages\FSharp.Charting.0.90.6"
#load "FSharp.Charting.fsx"


open FSharp.Charting

// ------------- CON ANDREA ----------------
// M matrice misurazioni
// x surrogato
// b misure target program
// t target measure
// T misure target rei programmi benchamerk
// t = T x
// M x = b
// x = M^{-1} b
// M = U S V*
// pseudoinversa di M = M* = U* S* V
// t = T x = T U* S* V b

let pseudoinverse (M:LinearAlgebra.Generic.Matrix<float>) =

    //    let A = DenseMatrix.CreateRandom(2,3,distr)
    //    let Ainv = A.Transpose() * (A * A.Transpose()).Inverse() 
    //    A * Ainv * A - A

    let D = M.Svd(true)
    let W = D.W()
    let s = D.S()
    let tolerance = Precision.EpsilonOf(2.0) * float(Math.Max(M.RowCount, M.ColumnCount)) * W.[0, 0]
    for i = 0 to s.Count-1 do
        s.[i] <- if s.[i] <= tolerance then 0.0 else 1.0 / s.[i]
    W.SetDiagonal(s)
    (D.U() * W * D.VT()).Transpose()


let contains (ary: int array) (value: int) =
  ary |> Array.map (fun v -> v = value) |> Array.exists (id)

let percentile (p:float) (data:float[]) =
    let sorted = data |> Array.map abs |> Array.sort
    let idx = int(float(sorted.Length) * p)
    sorted.[idx]

let percentiles data =
    percentile 0.25 data, percentile 0.5 data, percentile 0.75 data




// -------------------------------------- CHECK MODEL ---------------------------------------
(*
let distr = MathNet.Numerics.Distributions.ContinuousUniform()
let distr0 = MathNet.Numerics.Distributions.ContinuousUniform(-1.0, 1.0)
let distr1 = MathNet.Numerics.Distributions.ContinuousUniform(0.0, 2.0)

//  ----- same program different size ----
let referenceMeasures = DenseMatrix.CreateRandom(4, 1, distr)
let hiddenMeasures = DenseMatrix.CreateRandom(3, 1, distr)
let referencePrograms = DenseMatrix.CreateRandom(1, 5, distr)
let targetProgram = DenseMatrix.CreateRandom(1, 1, distr)
let A = referenceMeasures * referencePrograms
let a = hiddenMeasures * referencePrograms
let b = referenceMeasures * targetProgram

let noiseAmount = 0.10
let A1 = A + DenseMatrix.CreateRandom(A.RowCount, A.ColumnCount, distr0) * noiseAmount
let a1 = a + DenseMatrix.CreateRandom(a.RowCount, a.ColumnCount, distr0) * noiseAmount
let b1 = b + DenseMatrix.CreateRandom(b.RowCount, b.ColumnCount, distr0) * noiseAmount
//A.SetRow(0, DenseVector.Create(A.ColumnCount, fun _ -> 3.5))
//A.SetRow(2, 1.0 * DenseVector.CreateRandom(A.ColumnCount, distr))
//b.SetRow(0, 20.0 * DenseVector.CreateRandom(b.ColumnCount, distr))
let inv = pseudoinverse A
let predict = a * inv
hiddenMeasures * targetProgram
predict * b

let inline cons a b = (a,b)


//  ----- independent programs ----

let benchmarks = 4
let resources = 4
let A = DenseMatrix.CreateRandom(resources, benchmarks, distr)
let makeExponential (A:DenseMatrix) =
    for i in 0..(A.ColumnCount - 1) do
        A.SetColumn(i, A.Column(i) * Math.Pow(10.0, float(i)) )
makeExponential A
let a = DenseMatrix.CreateRandom(10000, benchmarks, distr)
makeExponential a

let one = Func<int,int,float>(fun i j -> 1.0)
let noiseAmount = 0.001
let A1 = A.PointwiseMultiply(DenseMatrix.CreateRandom(A.RowCount, A.ColumnCount, distr0) * noiseAmount + DenseMatrix.Create(A.RowCount, A.ColumnCount, one)) //*0.0 // * 0.0 means no noise here
let a1 = a.PointwiseMultiply(DenseMatrix.CreateRandom(a.RowCount, a.ColumnCount, distr0) * noiseAmount + DenseMatrix.Create(a.RowCount, a.ColumnCount, one)) //*0.0     // noise here
let b1 = b.PointwiseMultiply(DenseMatrix.CreateRandom(b.RowCount, b.ColumnCount, distr0) * noiseAmount + DenseMatrix.Create(b.RowCount, b.ColumnCount, one)) //*0.0 // no noise here

let x = DenseMatrix.CreateRandom(benchmarks, 1, distr)
let exact = a * x

let A0 = A1.Clone()
let a0 = a1.Clone()
let b0 = b1.Clone()
let x0 = x.Clone()




let permIndices = Array.init (A.ColumnCount) (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst

//let permIndices = [| 0; 1; 2; 3 |]

let A2 = A0.Clone()
let a2 = a0.Clone()
let b2 = b0.Clone()
let x2 = x0.Clone()

A2.PermuteColumns(new Permutation(permIndices))
a2.PermuteColumns(new Permutation(permIndices))
x2.PermuteRows(new Permutation(permIndices))

//let b2 = A2 * x2
let s = sprintf "%A" permIndices

let inv = pseudoinverse A2
let predict = a2 * inv
let estimate = predict * b2
estimate.PointwiseDivide(exact).Column(0) |> Seq.map (fun v -> abs (v - 1.0)) |> Seq.sort |> Seq.take 8000 |> Seq.toArray |> Array.mapi cons |> Chart.Line |> Chart.WithTitle s


*)



// -------------------- NMF -------------------------


#r @"X:\Projects\EnergonWin8\Solvers\bin\Debug\Solvers.dll"
open Energon.Solvers

let NMFSolver = Energon.Solvers.predictUsingNonNegativeMatrixFactorization


type Result = {
 M: DenseMatrix;
 p1 : float;
 p2: float;
 p3: float;
 m1: float;
 m2: float;
 m3: float;
 cond: float;
}

let inline cons a b = (a,b)

let contains (ary) (value) =
  ary |> Array.map (fun v -> v = value) |> Array.exists (id)

let randomizedTargetBasis targetAlg basisSize =
    // filter out cases with work size too little, as .net overhead is larger than actual completion time
    let lowerWorksize = 0.0//1024.0//*1024.0
    let filtered = workSize |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> v >= lowerWorksize) |> Array.map fst
    // take all the algorithm ids of each case, filter out cases with workload below lower limit
    let filteredIdxAlg = alg |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> contains filtered i)|> Seq.toArray
    // split cases into belonging to target algorithm or not
    let casesTargetAlg = filteredIdxAlg |> Array.filter (fun (i,v) -> v=targetAlg) |> Array.map fst
    let casesNoTargetAlg = filteredIdxAlg |> Array.filter (fun (i,v) -> v <> targetAlg || true) |> Array.map fst
    // pick a target case belonging to target algorithm
    let target = casesTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.head
    // remove the target from the algorithms used in the basis (makes sense only if target alg is allowed in the base)
    let casesNoTargetAlgNoTarget = casesNoTargetAlg |> Array.filter (fun i -> i <> target)
    // pick a random base to build the model with
    let basis = casesNoTargetAlgNoTarget |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.take basisSize |> Seq.toArray
    // TMP: ONLY USE CASES FROM THE TARGET ALG
    //let basis = casesTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.take basisSize |> Seq.toArray
    let basis = casesTargetAlg //|> Seq.take (casesTargetAlg.Length - 1) |> Seq.toArray
    //let basis = filteredIdxAlg |> Array.map fst
    target, basis

let normColumn (col:float array) =
  let m = Array.max col
  col |> Array.map (fun v -> v/m)
(*
// zero centered
let normColumn col =
  let m = Array.average col
  let col0 = col |> Array.map (fun v -> v - m)
  let std = col0 |> Array.map sqr |> Array.average |> sqrt
  col0 |> Array.map ((*) (1.0 / std))

// not zero centered
let normColumn col =
  let m = Array.average col
  let col0 = col |> Array.map (fun v -> v - m)
  let std = col0 |> Array.map sqr |> Array.average |> sqrt
  col |> Array.map ((*) (1.0 / std))
*)


/// generates a random selection of programs for an experiment 
/// returns:
/// a random target program in a specified family of algorithms
/// a random basis (list of programs), in a specified list of possible families of algorithms
/// the known measures of the basis (A)
/// the known measures of the target program (b)
/// the hidden measures of the basis (a)
/// the hidden measures of the target program: the measure we want to predict
let generateRandomCase (knownMeasures:DenseMatrix) (hiddenMeasures:DenseMatrix) targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize =
    // the list of indices of possible targets
    let casesTargetAlg = alg |> Array.mapi cons |> Array.filter (fun (i,v) -> v = targetAlgFamily) |> Array.map fst
    // the list of indices of possible programs for the basis 
    //let casesBasisAlgs = alg |> Array.mapi cons |> Array.filter (fun (i,v) -> contains allowedAlgFamiliesInBasis v) |> Array.map fst
    let casesBasisAlgs = allowedAlgFamiliesInBasis |> Seq.map (fun sel -> alg |> Array.mapi cons |> Array.filter (fun (i,v) -> v = sel) |> Array.map fst)
    let target = casesTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.head
    //let basis = casesBasisAlgs |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.truncate maxBasisSize |> Seq.sortBy (id) // |> Seq.sortBy (fun _ -> rnd.NextDouble())
    let basis = casesBasisAlgs |> Seq.map (fun cases -> cases |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.truncate maxBasisSize ) |> Seq.concat |> Seq.sort  |> Seq.sortBy (fun _ -> rnd.NextDouble())  |> Seq.toArray 

    let b =  knownMeasures.Column(target).ToColumnMatrix()
    let A = DenseMatrix.OfColumns(knownMeasures.RowCount, basis |> Seq.length, basis |> Seq.map (fun i -> knownMeasures.Column(i) |> Seq.map id ) )
    let a = DenseMatrix.OfColumns(hiddenMeasures.RowCount, basis |> Seq.length, basis |> Seq.map (fun i -> hiddenMeasures.Column(i) |> Seq.map id ) )
    let secret = hiddenMeasures.Column(target).ToColumnMatrix()
    target, basis, A, b, a, secret 

/// does one prediction
let predictCase targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize = 
       
    let normColumn (x:_[]) = Array.map2 (/) x integratedGPUTime
    let knownMeasures = DenseMatrix.OfRows(3,arithmetic.Length, [| 
                                                            dataTrasfFromHost |> Array.map (fun v -> 1.0) |> normColumn; 
                                                            dataTrasfFromDevice|> normColumn; 
                                                            //readAccessGlobal ; 
                                                            //simple_op |> normColumn; 
                                                            complex_op |> normColumn  (*; derived *) 
                                                            |])
    let hiddenMeasures = DenseMatrix.OfRows(3,arithmetic.Length, [| 
                                                            discreteGPUTime |> normColumn; 
                                                            integratedGPUTime |> normColumn; 
                                                            CPUTime |> normColumn  |])
    let target, basis, A, b, a, secret = generateRandomCase knownMeasures hiddenMeasures targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize
//    let W,H,d = Energon.Solvers.NonNegativeMatrixFactorization A 5
//    let Wstar = pseudoinverse W
//    let Hstar = pseudoinverse H
//    let h = a * Wstar
//    let w = Hstar * b
//    let predicted = h * w
    let predict = a * pseudoinverse A
    let predicted = predict * b
    if false then
        printfn "basis: %A" basis
        printfn "M: %A" A
        printfn "predict: %A" predict
        printfn "relative error on M: %A" <| (predict * A).PointwiseDivide(a)
        for i = 0 to predict.RowCount-1 do
            printfn "%A" <| (predict.Row(i).ToColumnMatrix() * DenseMatrix.Create(1, A.ColumnCount, fun i j -> 1.0)).PointwiseMultiply(A)            
    let measured = secret
    //let denorm = integratedGPUTime.[target]
    let err = //(predicted - measured).Column(0) |> Seq.toArray 
        predicted.PointwiseDivide(measured).Column(0) |> Seq.toArray
    err    

let target =3.0
let basis = [|  3.0; |]
let size = 200
let res = Array.init 100 (fun _ -> predictCase target basis size) |> Array.concat
let title = sprintf "%A %A %A" target basis size 
let inline cons a b = (a,b)
res |> Seq.map (fun v -> abs (v - 1.0)) |> Seq.sort |> Seq.truncate 100 |> Seq.toArray |> Array.mapi cons |> Chart.Line |> Chart.WithTitle title

//  ----------------------- predict best device ----------------------- 

/// does one prediction
let predictCase targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize = 
       
    let normColumn (x:_[]) = Array.map2 (/) x integratedGPUTime
    let knownMeasures = DenseMatrix.OfRows(3,arithmetic.Length, [| 
                                                            dataTrasfFromHost |> Array.map (fun v -> 1.0) |> normColumn; 
                                                            dataTrasfFromDevice|> normColumn; 
                                                            //readAccessGlobal ; 
                                                            //simple_op |> normColumn; 
                                                            complex_op |> normColumn  (*; derived *) 
                                                            |])
    let hiddenMeasures = DenseMatrix.OfRows(2,arithmetic.Length, [| 
                                                            discreteGPUTime |> normColumn; 
                                                            //integratedGPUTime |> normColumn; 
                                                            CPUTime |> normColumn  |])
    let target, basis, A, b, a, secret = generateRandomCase knownMeasures hiddenMeasures targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize
//    let W,H,d = Energon.Solvers.NonNegativeMatrixFactorization A 5
//    let Wstar = pseudoinverse W
//    let Hstar = pseudoinverse H
//    let h = a * Wstar
//    let w = Hstar * b
//    let predicted = h * w
    let predict = a * pseudoinverse A
    let predicted = predict * b
    if false then
        printfn "basis: %A" basis
        printfn "M: %A" A
        printfn "predict: %A" predict
        printfn "relative error on M: %A" <| (predict * A).PointwiseDivide(a)
        for i = 0 to predict.RowCount-1 do
            printfn "%A" <| (predict.Row(i).ToColumnMatrix() * DenseMatrix.Create(1, A.ColumnCount, fun i j -> 1.0)).PointwiseMultiply(A)            
    let measured = secret
    let err = //(predicted - measured).Column(0) |> Seq.toArray 
        predicted.PointwiseDivide(measured).Column(0) |> Seq.toArray
        

    let denorm = integratedGPUTime.[target]
    let realPredictions = predicted.Multiply(denorm)
    let realMeasures = measured.Multiply(denorm)
    let minIndex = realPredictions.Column(0).ToArray() |> Array.mapi cons |> Array.sortBy snd |> Seq.head |> fst
    let minIndexReal = realMeasures.Column(0).ToArray() |> Array.mapi cons |> Array.sortBy snd |> Seq.head |> fst
    minIndex = minIndexReal, err, realPredictions, realMeasures



let target = 5.0
let basis = [| 1.0; 3.0 |]
let size = 200
let res = Array.init 100 (fun _ -> predictCase target basis size) 
let yeah = res |> Array.map (fun (b,_,_,_) -> b) |> Array.filter (fun v -> v) |> Array.length
let success = float(yeah)/float(res.Length)



// ----------------- study predictions ---------------------



/// does one prediction
let predictCase targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize = 
       
    let normColumn (x:_[]) = Array.map2 (/) x integratedGPUTime
    let knownMeasures = DenseMatrix.OfRows(3,arithmetic.Length, [| 
                                                            dataTrasfFromHost |> Array.map (fun v -> 1.0) |> normColumn; 
                                                            dataTrasfFromDevice|> normColumn; 
                                                            //readAccessGlobal ; 
                                                            //simple_op |> normColumn; 
                                                            complex_op |> normColumn  (*; derived *) 
                                                            |])
    let hiddenMeasures = DenseMatrix.OfRows(2,arithmetic.Length, [| 
                                                            discreteGPUTime |> normColumn; 
                                                            //integratedGPUTime |> normColumn; 
                                                            CPUTime |> normColumn  |])
    let target, basis, A, b, a, secret = generateRandomCase knownMeasures hiddenMeasures targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize
//    let W,H,d = Energon.Solvers.NonNegativeMatrixFactorization A 5
//    let Wstar = pseudoinverse W
//    let Hstar = pseudoinverse H
//    let h = a * Wstar
//    let w = Hstar * b
//    let predicted = h * w
    let predict = a * pseudoinverse A
    let predicted = predict * b
    if false then
        printfn "basis: %A" basis
        printfn "M: %A" A
        printfn "predict: %A" predict
        printfn "relative error on M: %A" <| (predict * A).PointwiseDivide(a)
        for i = 0 to predict.RowCount-1 do
            printfn "%A" <| (predict.Row(i).ToColumnMatrix() * DenseMatrix.Create(1, A.ColumnCount, fun i j -> 1.0)).PointwiseMultiply(A)            
    let measured = secret
    let err = //(predicted - measured).Column(0) |> Seq.toArray 
        predicted.PointwiseDivide(measured).Column(0) |> Seq.toArray
        

    let denorm = integratedGPUTime.[target]
    let realPredictions = predicted.Multiply(denorm)
    let realMeasures = measured.Multiply(denorm)
    let minIndex = realPredictions.Column(0).ToArray() |> Array.mapi cons |> Array.sortBy snd |> Seq.head |> fst
    let minIndexReal = realMeasures.Column(0).ToArray() |> Array.mapi cons |> Array.sortBy snd |> Seq.head |> fst
    minIndex = minIndexReal, err, realPredictions, realMeasures


let res = Array.init 1000 (fun _ -> let order, err, pred, meas = predictCase 5.0 [| 5.0 |] 10000 in abs(pred.[1,0] / meas.[1,0] - 1.0))
res |> Array.sort |> Array.mapi cons |> Chart.Line









// ------------------ work on a specific case ------------------


let generateFixedCase (knownMeasures:LinearAlgebra.Generic.Matrix<float>) (hiddenMeasures:LinearAlgebra.Generic.Matrix<float>) target basis =
    let b =  knownMeasures.Column(target).ToColumnMatrix()
    let A = DenseMatrix.OfColumns(knownMeasures.RowCount, basis |> Seq.length, basis |> Seq.map (fun i -> knownMeasures.Column(i) |> Seq.map id ) )
    let a = DenseMatrix.OfColumns(hiddenMeasures.RowCount, basis |> Seq.length, basis |> Seq.map (fun i -> hiddenMeasures.Column(i) |> Seq.map id ) )
    let secret = hiddenMeasures.Column(target).ToColumnMatrix()
    target, basis, A, b, a, secret 
let predictFixedCase target basis =    
    let knownMeasures = DenseMatrix.OfRows(3,arithmetic.Length, [| 
                                                            dataTrasfFromHost |> Array.map (fun v -> 1.0); 
                                                            dataTrasfFromDevice ; 
                                                            //readAccessGlobal ; 
                                                            //simple_op ; 
                                                            complex_op   (*; derived *) 
                                                            |])
    let knownMeasures = knownMeasures.PointwiseDivide(DenseMatrix.OfRows(3,arithmetic.Length, [| integratedGPUTime ;
                                                          integratedGPUTime ;
                                                          integratedGPUTime
                                                        |]))
    let hiddenMeasures = DenseMatrix.OfRows(1,arithmetic.Length, [| //discreteGPUTime (*|> normColumn*); 
                                                            //integratedGPUTime (*|> normColumn*); 
                                                            CPUTime (*|> normColumn *) |])
    let hiddenMeasures = hiddenMeasures.PointwiseDivide(DenseMatrix.OfRows(1,arithmetic.Length, [|
                                                                                                    integratedGPUTime
                                                                                                |]))
    let target, basis, A, b, a, secret = generateFixedCase knownMeasures hiddenMeasures target basis
//    let W,H,d = Energon.Solvers.NonNegativeMatrixFactorization M 5
//    let Wstar = pseudoinverse W
//    let Hstar = pseudoinverse H
//    let h = a * Wstar
//    let w = Hstar * b
//    let predicted = h * w
    let predict = a * pseudoinverse A
    if false then
        printfn "target: %A" target
        printfn "basis: %A" basis
        printfn "A: %A" A
        printfn "x: %A" ((pseudoinverse A) * b)
        printfn "predict: %A" predict
        printfn "relative error on M: %A" <| (predict * A).PointwiseDivide(a)
        for i = 0 to predict.RowCount-1 do
            printfn "%A" <| (predict.Row(i).ToColumnMatrix() * DenseMatrix.Create(1, A.ColumnCount, fun i j -> 1.0)).PointwiseMultiply(A)            
    let predicted = predict * b
    let measured = secret
    let err = //(predicted - measured).Column(0) |> Seq.toArray 
        predicted.PointwiseDivide(measured).Column(0) |> Seq.toArray
    //err
    Seq.map2 cons (predicted.Column(0)) (measured.Column(0)) |> Seq.toArray

let res = Array.init 14 (fun i -> predictFixedCase (0+i) (Array.init 14 (fun j->0+j)) ) |> Array.concat

let res = Array.init 54 (fun i -> predictFixedCase (78+i) (Array.init 54 (fun j->78+j)) ) |> Array.concat
let res = Array.init 54 (fun i -> predictFixedCase (78+i) [| 80 ; 130; 100 |] ) |> Array.concat
let res = Array.init 45 (fun i -> predictFixedCase i [| 0; 7; 13; 14; 40 |] ) |> Array.concat
let inline cons a b = (a,b)
res |> Seq.map (fun (p,m) -> p/m-1.0) |> Seq.toArray |> Array.mapi cons |> Chart.Line



// ------------- try to find a case where permutation changes prediction ------------------

let generateRandomCase2 targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize =
    // the list of indices of possible targets
    let casesTargetAlg = alg |> Array.mapi cons |> Array.filter (fun (i,v) -> v = targetAlgFamily) |> Array.map fst
    // the list of indices of possible programs for the basis 
    //let casesBasisAlgs = alg |> Array.mapi cons |> Array.filter (fun (i,v) -> contains allowedAlgFamiliesInBasis v) |> Array.map fst
    let casesBasisAlgs = allowedAlgFamiliesInBasis |> Seq.map (fun sel -> alg |> Array.mapi cons |> Array.filter (fun (i,v) -> v = sel) |> Array.map fst)
    let target = casesTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.head
    //let basis = casesBasisAlgs |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.truncate maxBasisSize |> Seq.sortBy (id) // |> Seq.sortBy (fun _ -> rnd.NextDouble())
    let basis = casesBasisAlgs |> Seq.map (fun cases -> cases |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.truncate maxBasisSize ) |> Seq.concat |> Seq.sort |> Seq.toArray
    let basisPermutated = basis |> Seq.sortBy (fun _ -> rnd.NextDouble()) |> Seq.toArray
    target, basis, basisPermutated


let predictCase2 targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize =    
    let normColumn (x:_[]) = Array.map2 (/) x integratedGPUTime
    let knownMeasures = DenseMatrix.OfRows(3,arithmetic.Length, [| 
                                                            dataTrasfFromHost |> Array.map (fun v -> 1.0) |> normColumn; 
                                                            dataTrasfFromDevice|> normColumn; 
                                                            //readAccessGlobal ; 
                                                            //simple_op |> normColumn; 
                                                            complex_op |> normColumn  (*; derived *) 
                                                            |])
    let hiddenMeasures = DenseMatrix.OfRows(1,arithmetic.Length, [| //discreteGPUTime (*|> normColumn*); 
                                                            //integratedGPUTime (*|> normColumn*); 
                                                            CPUTime |> normColumn  |])
    let target, basisOrdered, basisPermutated = generateRandomCase2 targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize
    let testOrdered    = (predictFixedCase target basisOrdered   ) |> Array.map (fun (x,y) -> x/y)
    let testPermutated = (predictFixedCase target basisPermutated) |> Array.map (fun (x,y) -> x/y) 
    if not (Seq.forall2 (fun a b -> abs(a/b-1.0)<1.e-4) testOrdered testPermutated ) then
        printfn "target %A" target
        printfn "basisOrdered %A" basisOrdered
        printfn "basisPermutated %A" basisPermutated
        printfn "errors ordered %A" testOrdered
        printfn "errore permutated %A" testPermutated

    true

let testFamily algFamily =
    Array.init 100 (fun size -> predictCase2 algFamily [| algFamily |] (1+size))
let testall () =
    Array.init 5 (fun i -> testFamily (float(1+i))) |> Array.concat |> Array.length

Array.init 1000000 (fun _ -> testall())

predictFixedCase 9 [| 9; 12 |]

// ---------------- try NMF on synthetic data ---------------- 

// H W = A
// h W = a
// H w = b
// h w = ?
// w = H* b
// h = a W*
// a W* H* b = ?
let distr = MathNet.Numerics.Distributions.ContinuousUniform()
let distr0 = MathNet.Numerics.Distributions.ContinuousUniform(-1.0, 1.0)

let one = Func<int,int,float>(fun i j -> 1.0)
// ------ hidden factors -----------
let benchmarks = 4
let resources = 4
let hiddenFactors = 7
let factorsCosts = DenseMatrix.CreateRandom(resources, hiddenFactors, distr)
let benchmarkComposition = DenseMatrix.CreateRandom(hiddenFactors, benchmarks, distr)
let A = factorsCosts * benchmarkComposition
let targetComposition = DenseMatrix.CreateRandom(hiddenFactors, 1, distr)
let factorsHiddenCosts = DenseMatrix.CreateRandom(1000, hiddenFactors, distr)
let a = factorsHiddenCosts * benchmarkComposition
let b = factorsCosts * targetComposition
let measured = factorsHiddenCosts * targetComposition

//  ----- independent benchmarks ----
//let A = DenseMatrix.CreateRandom(resources, benchmarks, distr)
////let A = DenseMatrix.Identity(4)
//let a = DenseMatrix.CreateRandom(1000, benchmarks, distr)
//let x = DenseMatrix.CreateRandom(benchmarks, 1, distr)
//// target program perfect linear combination
//let b = A * x
//let measured = a * x

// add some multiplicative noise
let noiseAmount = 0.1
let A1 = A.PointwiseMultiply(DenseMatrix.CreateRandom(A.RowCount, A.ColumnCount, distr0) * noiseAmount + DenseMatrix.Create(A.RowCount, A.ColumnCount, one)) //*0.0 // * 0.0 means no noise here
let a1 = a.PointwiseMultiply(DenseMatrix.CreateRandom(a.RowCount, a.ColumnCount, distr0) * noiseAmount + DenseMatrix.Create(a.RowCount, a.ColumnCount, one)) //*0.0     // noise here
let b1 = b.PointwiseMultiply(DenseMatrix.CreateRandom(b.RowCount, b.ColumnCount, distr0) * noiseAmount + DenseMatrix.Create(b.RowCount, b.ColumnCount, one)) //*0.0 // no noise here
// additive noise
//let A1 = A + (DenseMatrix.CreateRandom(A.RowCount, A.ColumnCount, distr0) * noiseAmount)  //*0.0 // * 0.0 means no noise here
//let a1 = a + (DenseMatrix.CreateRandom(a.RowCount, a.ColumnCount, distr0) * noiseAmount)  //*0.0     // noise here
//let b1 = b + (DenseMatrix.CreateRandom(b.RowCount, b.ColumnCount, distr0) * noiseAmount)  //*0.0 // no noise here


let A1 = DenseMatrix.OfMatrix(A1)

let W,H,d = Energon.Solvers.NonNegativeMatrixFactorization A1 7

// a W* H* b = ?
let Wstar = pseudoinverse W
let Hstar = pseudoinverse H
let h = a1 * Wstar
let w = Hstar * b1
let predicted = h * w
//(H * W) - A1
//(Wstar * Hstar) - (pseudoinverse A1)
//let predicted = a1 * (Wstar * Hstar) * b1
//let predicted = (a1 * Wstar) * (Hstar * b1)
//let predicted = a1 * (pseudoinverse A1) * b1
let errors = predicted.PointwiseDivide(measured).Column(0) |> Seq.map (fun v -> abs (v - 1.0)) |> Seq.sort |> Seq.take 800 |> Seq.toArray |> Array.mapi cons |> Chart.Line

let Aavg = A1.RowEnumerator() |> Seq.map (fun (i,v) -> v.Sum() / float(v.Count)) |> Seq.toArray
let aavg = a1.RowEnumerator() |> Seq.map (fun (i,v) -> v.Sum() / float(v.Count)) |> Seq.toArray

let A0 = DenseMatrix.OfRowVectors(A1.RowEnumerator() |> Seq.map (fun (i,v) -> v - Aavg.[i]) |> Seq.toArray)
let a0 = DenseMatrix.OfRowVectors(a1.RowEnumerator() |> Seq.map (fun (i,v) -> v - aavg.[i]) |> Seq.toArray)
let b0 = DenseMatrix.OfRowVectors(b1.RowEnumerator() |> Seq.map (fun (i,v) -> v - Aavg.[i]) |> Seq.toArray)


estimate.PointwiseDivide(exact).Column(0) |> Seq.map (fun v -> abs (v - 1.0)) |> Seq.sort |> Seq.toArray |> Array.mapi cons |> Chart.Line




// cond
let doManyAttempts target basissize =
    let attempt2 targetAlg basisSize =
        let target, basis = randomizedTargetBasis targetAlg basisSize
        let r = attempt target basis
        r.cond
    let res = Array.init 100 (fun _ -> attempt2 target basissize)
    let resm1 = Array.map 
    let avg = res |> Array.average
    let prod = res |> Array.map abs |> Array.reduce (*)
    let geomAverage = Math.Pow(prod, 1.0 / float(res.Length))
    let stddev = res |> Array.map (fun v -> (v-avg)*(v-avg)) |> Array.average |> sqrt
    (avg, stddev, res.Length) //, stddev
    //geomAverage
let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (1 + 2*measures)) )


// right prediction
// TODO DOES NOT MAKE SENSE IF I NORMALIZE COMPLETION TIMES INDIVIDUALLY
let doManyAttempts target basissize =
    let attempt2 targetAlg basisSize =
        let target, basis = randomizedTargetBasis targetAlg basisSize
        printf "%i %A\n" target basis
        let r = attempt target basis
        let predictions = [| 0,r.p3; 1,r.p1; 2,r.p2 |]
        let measures = [| 0,r.m3; 1,r.m1; 2,r.m2 |]
        let bestPredicted,_ = predictions |> Array.sortBy (fun (i,v) -> v) |> Seq.head
        let bestMeasured,_ = measures |> Array.sortBy (fun (i,v) -> v) |> Seq.head
        bestPredicted = bestMeasured

    let res = Array.init 100 (fun _ -> attempt2 target basissize) 
    let ok = res |> Array.filter (fun b -> b) |> Array.length
    float(ok)/float(res.Length)
let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (6 + 2*measures)) )

let doManyAttempts target basissize =
    let res = Array.init 100 (fun _ -> attempt target basissize) |> Array.filter isfinite
    let avg = res |> Array.average
    let prod = res |> Array.map abs |> Array.reduce (*)
    let geomAverage = Math.Pow(prod, 1.0 / float(res.Length))
    let stddev = res |> Array.map (fun v -> (v-avg)*(v-avg)) |> Array.average |> sqrt
    (avg, stddev, res.Length) //, stddev
    //geomAverage
let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (1 + 2*measures)) )

Array.init 1000 (fun _ -> attempt 1.0 6) |> Array.filter isfinite |> Array.sort |> Array.mapi (fun i v -> (i,v)) |> Chart.Line

let doManyAttempts target basissize =
    let res = Array.init 100 (fun _ -> attempt target basissize) |> Array.map (fun (v1,v2,v3,c) -> [| v1; v2; v3 |]) |> Array.concat
    //let res = res |> Array.map sqr
    let avg = res |> Array.average
    let prod = res |> Array.map abs |> Array.reduce (*)
    let geomAverage = Math.Pow(prod, 1.0 / float(res.Length))
    let stddev = res |> Array.map (fun v -> (v-avg)*(v-avg)) |> Array.average |> sqrt
    avg, stddev
    //geomAverage

let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (1 + 2*measures)) )

let r1,r2,r3 = Array.init 1000 (fun i -> attempt 3.0 20) |> Array.unzip3

let percentile (p:float) (data:float[]) =
    let sorted = data |> Array.map abs |> Array.sort
    let idx = int(float(sorted.Length) * p)
    sorted.[idx]

let percentiles data =
    percentile 0.25 data, percentile 0.5 data, percentile 0.75 data

let doCase alg size = 
    Array.init 1000 (fun i -> attempt alg size) |> Array.map (fun (a,b,c) -> [|a;b;c|]) |> Array.concat
let r = doCase 4.0 20

percentiles r

A.ToMatrixString(2,3)

let writeCSV (M:LinearAlgebra.Generic.Matrix<float>) filename =
    let sb = new System.Text.StringBuilder()
    let writeRow (row: float array) = 
        row |> Array.iter (fun v -> sb.AppendFormat(@"{0} ", v) |> ignore)
        sb.AppendLine() |> ignore
    M.RowEnumerator() |> Seq.iter (fun (i,row) -> writeRow (row |> Seq.toArray))
    System.IO.File.WriteAllText(filename, sb.ToString())


let knownMeasures = DenseMatrix.OfRows(3,arithmetic.Length, [| 
                                                        dataTrasfFromHost |> Array.map (fun v -> 1.0); 
                                                        dataTrasfFromDevice ; 
                                                        //readAccessGlobal ; 
                                                        //simple_op ; 
                                                        complex_op   (*; derived *) 
                                                        |])
let knownMeasures = knownMeasures.PointwiseDivide(DenseMatrix.OfRows(3,arithmetic.Length, [| integratedGPUTime ;
                                                        integratedGPUTime ;
                                                        integratedGPUTime
                                                    |]))
let hiddenMeasures = DenseMatrix.OfRows(1,arithmetic.Length, [| //discreteGPUTime (*|> normColumn*); 
                                                        //integratedGPUTime (*|> normColumn*); 
                                                        CPUTime (*|> normColumn *) |])
let hiddenMeasures = hiddenMeasures.PointwiseDivide(DenseMatrix.OfRows(1,arithmetic.Length, [|
                                                                                                integratedGPUTime
                                                                                            |]))
let target, basis, A, b, a, secret = generateFixedCase knownMeasures hiddenMeasures 1 [| 14; 20; 23 |]


writeCSV A @"X:\Projects/featureBasedScheduling/AnalyseData/octave/A.txt"

let svd = A.Svd(true)
svd.S()
svd.U()
pseudoinverse A

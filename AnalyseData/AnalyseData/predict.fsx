open System

#I @"U:\Projects\EnergonWin8\packages\FSharp.Charting.0.90.6"
#load "FSharp.Charting.fsx"


open FSharp.Charting

// ----------- Load data --------------

let datafolder = @"U:\Projects\featureBasedScheduling\Dati"

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

#r @"U:\Projects\featureBasedScheduling\AnalyseData\packages\MathNet.Numerics.FSharp.2.6.0\lib\net40\MathNet.Numerics.FSharp.dll"
#I @"U:\Projects\featureBasedScheduling\AnalyseData\packages\MathNet.Numerics.2.6.2\lib\net40"
#r @"MathNet.Numerics.dll"
#r @"MathNet.Numerics.IO.dll"
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra.Double

//let target = 2

//let basisSize = 2

let rnd = new System.Random()

let mat (rows:(float seq) array) = DenseMatrix.OfRows(rows |> Seq.length ,rows.[0] |> Seq.length, rows)




// ------------------------------------------------------------------------------


#I @"U:\Projects\EnergonWin8\packages\FSharp.Charting.0.90.6"
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

let inline cons a b = (a,b)


(*
// even odd rule

/// generates a random selection of programs for an experiment 
/// returns:
/// a random target program in a specified family of algorithms
/// a random basis (list of programs), in a specified list of possible families of algorithms
/// the known measures of the basis (A)
/// the known measures of the target program (b)
/// the hidden measures of the basis (a)
/// the hidden measures of the target program: the measure we want to predict
let generateRandomCase (knownMeasures:LinearAlgebra.Generic.Matrix<float>) (hiddenMeasures:LinearAlgebra.Generic.Matrix<float>) targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize =
    // the list of indices of possible targets
    let heads = if rnd.NextDouble() < 0.5 then 0 else 1
    let casesTargetAlg = alg |> Array.mapi cons |> Array.filter (fun (i,v) -> v = targetAlgFamily && i % 2 = heads) |> Array.map fst
    // the list of indices of possible programs for the basis 
    //let casesBasisAlgs = alg |> Array.mapi cons |> Array.filter (fun (i,v) -> contains allowedAlgFamiliesInBasis v) |> Array.map fst
    let casesBasisAlgs = allowedAlgFamiliesInBasis |> Seq.map (fun sel -> alg |> Array.mapi cons |> Array.filter (fun (i,v) -> v = sel && i % 2 <> heads) |> Array.map fst)
    let target = casesTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.head
    //let basis = casesBasisAlgs |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.truncate maxBasisSize |> Seq.sortBy (id) // |> Seq.sortBy (fun _ -> rnd.NextDouble())
    let basis = casesBasisAlgs |> 
                Seq.map (fun cases -> cases |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.truncate maxBasisSize ) |> 
                Seq.concat |> Seq.sort  |> Seq.sortBy (fun _ -> rnd.NextDouble())  |> Seq.toArray 

    let b =  knownMeasures.Column(target).ToColumnMatrix()
    let A = DenseMatrix.OfColumns(knownMeasures.RowCount, basis |> Seq.length, basis |> Seq.map (fun i -> knownMeasures.Column(i) |> Seq.map id ) )
    let a = DenseMatrix.OfColumns(hiddenMeasures.RowCount, basis |> Seq.length, basis |> Seq.map (fun i -> hiddenMeasures.Column(i) |> Seq.map id ) )
    let secret = hiddenMeasures.Column(target).ToColumnMatrix()
    target, basis, A, b, a, secret 
*)

// leave one out
let generateRandomCase (knownMeasures:LinearAlgebra.Generic.Matrix<float>) (hiddenMeasures:LinearAlgebra.Generic.Matrix<float>) targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize =
    // the list of indices of possible targets
    let casesTargetAlg = alg |> Array.mapi cons |> Array.filter (fun (i,v) -> v = targetAlgFamily ) |> Array.map fst
    let target = casesTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.head
    // the list of indices of possible programs for the basis 
    let casesBasisAlgs = allowedAlgFamiliesInBasis |> Seq.map (fun sel -> alg |> Array.mapi cons |> Array.filter (fun (i,v) -> v = sel && i <> target) |> Array.map fst)
    let basis = casesBasisAlgs |> 
                Seq.map (fun cases -> cases |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.truncate maxBasisSize ) |> 
                Seq.concat |> Seq.sort  |> Seq.sortBy (fun _ -> rnd.NextDouble())  |> Seq.toArray 

    let b =  knownMeasures.Column(target).ToColumnMatrix()
    let A = DenseMatrix.OfColumns(knownMeasures.RowCount, basis |> Seq.length, basis |> Seq.map (fun i -> knownMeasures.Column(i) |> Seq.map id ) )
    let a = DenseMatrix.OfColumns(hiddenMeasures.RowCount, basis |> Seq.length, basis |> Seq.map (fun i -> hiddenMeasures.Column(i) |> Seq.map id ) )
    let secret = hiddenMeasures.Column(target).ToColumnMatrix()
    target, basis, A, b, a, secret 


// ----------------- study predictions ---------------------


let mutable print = false

/// does one prediction
let predictCase targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize = 
       
    //let normColumn (x:_[]) = Array.map2 (/) x (dataTrasfFromHost |> Array.map (fun v -> 1.0))
    let knownMeasures = mat [| dataTrasfFromHost |> Array.map (fun v -> 1.0); 
                               dataTrasfFromDevice; 
                               //readAccessGlobal ; 
                               simple_op; 
                               complex_op   (*; derived *) ;
                               //integratedGPUTime ;
                               |]

    let norms = knownMeasures.ColumnEnumerator() |> Seq.map (fun (i,col) -> 0.0 + 0.0 * discreteGPUTime.[i] + 1.0 * CPUTime.[i] + 0.0 * integratedGPUTime.[i] + 0.0 * col.Norm(2.0) ) |> Seq.toArray
    let knownMeasures = knownMeasures.PointwiseDivide(mat (Array.init (knownMeasures.RowCount) (fun  _ -> norms |> Seq.map (id))))
    let hiddenMeasures = mat [| 
                               discreteGPUTime ; 
                               integratedGPUTime ; 
                               CPUTime   
                              |]

    let hiddenMeasures = hiddenMeasures.PointwiseDivide(mat (Array.init (hiddenMeasures.RowCount) (fun  _ -> norms |> Seq.map (id))))
    let target, basis, A, b, a, secret = generateRandomCase knownMeasures hiddenMeasures targetAlgFamily allowedAlgFamiliesInBasis maxBasisSize
//    let W,H,d = Energon.Solvers.NonNegativeMatrixFactorization A 5
//    let Wstar = pseudoinverse W
//    let Hstar = pseudoinverse H
//    let h = a * Wstar
//    let w = Hstar * b
//    let predicted = h * w
    let predict = a * pseudoinverse A
    let predicted = predict * b
    if print then
        printfn "basis: %A" basis
        printfn "M: %A" A
        printfn "knownMeasures: %A" knownMeasures
        //printfn "predict: %A" predict
        //printfn "relative error on M: %A" <| (predict * A).PointwiseDivide(a)
        //for i = 0 to predict.RowCount-1 do
        //    printfn "%A" <| (predict.Row(i).ToColumnMatrix() * DenseMatrix.Create(1, A.ColumnCount, fun i j -> 1.0)).PointwiseMultiply(A)            
    let measured = secret
    let err = //(predicted - measured).Column(0) |> Seq.toArray 
        predicted.PointwiseDivide(measured).Column(0) |> Seq.toArray
        
    let realPredictions = predicted
    let realMeasures = measured
    let minIndex = realPredictions.Column(0).ToArray() |> Array.mapi cons |> Array.sortBy snd |> Seq.head |> fst
    let minIndexReal = realMeasures.Column(0).ToArray() |> Array.mapi cons |> Array.sortBy snd |> Seq.head |> fst
    minIndex = minIndexReal, err, realPredictions, realMeasures


let saveRes (res:float array) (filename:string) =
    let sb = new System.Text.StringBuilder()
    res |> Array.iter (fun (v:float) -> sb.AppendLine(String.Format(@"{0}", v)) |> ignore)
    let path = String.Format(@"U:\Projects\featureBasedSchedulingPaper\data\{0}", filename)
    System.IO.File.WriteAllText(path, sb.ToString())


let target = 5.0
let basis = [| 1.0; 2.0;  4.0; 5.0 |]
//predictCase target basis 100

let chartData res = res |> Array.sort |> Seq.truncate  10000 |> Seq.toArray |> Array.mapi cons |> Chart.Line

let res = Array.init 1000 (fun _ -> let order, err, pred, meas = predictCase target basis 100 in abs(pred.[0,0] / meas.[0,0] - 1.0), abs(pred.[1,0] / meas.[1,0] - 1.0), abs(pred.[2,0] / meas.[2,0] - 1.0) )
//let res = Array.init 1000 (fun _ -> let order, err, pred, meas = predictCase 5.0 [|  5.0; |] 100 in (pred.[1,0] / meas.[1,0] - 1.0))
Chart.Combine [| 
                //res |> Array.map (fun (a,b,c) -> a) |> chartData |> Chart.WithLegend(Title="Puppa")
                //res |> Array.map (fun (a,b,c) -> b) |> chartData;
                res |> Array.map (fun (a,b,c) -> c) |> chartData;
              |]


let basisStr = 
    let sb = new System.Text.StringBuilder()
    basis |> Array.iter (fun f -> sb.AppendFormat(@"{0}", int(f)) |> ignore)
    sb.ToString()
let file = String.Format(@"errors_target{0}_basis{1}_featuresAll.dat", int(target), basisStr)
saveRes res file





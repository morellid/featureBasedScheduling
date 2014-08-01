open System

#I @"X:\Projects\EnergonWin8\packages\FSharp.Charting.0.90.6"
#load "FSharp.Charting.fsx"


open FSharp.Charting

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


// best device prediction
//let targetAlg =  2.0

let attempt targetAlg basisSize =
    // filter out cases with work size too little, as .net overhead is larger than actual completion time
    let lowerWorksize = 1024.0*1024.0
    let contains (ary: int array) (value: int) =
        ary |> Array.map (fun v -> v = value) |> Array.exists (id)
    let filtered = workSize |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> v >= lowerWorksize) |> Array.map fst
    let filteredAlg = alg |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> contains filtered i) |> Seq.map snd |> Seq.toArray
    // split cases into belonging to target algorithm or not
    let casesTargetAlg = filteredAlg |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> v=targetAlg) |> Array.map fst
    // pick a target case belonging to target algorithm
    let target = casesTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.head
    let casesNoTargetAlg = filteredAlg |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> (*v <> targetAlg*) true) |> Array.map fst
    // pick a random base to build the model with
    let basis = casesNoTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.take basisSize |> Seq.toArray
    let M_all = DenseMatrix.OfRows(5,arithmetic.Length, [| dataTrasfFromHost |> normColumn; 
                                                            dataTrasfFromDevice |> normColumn; 
                                                            readAccessGlobal |> normColumn; 
                                                            simple_op |> normColumn; 
                                                            complex_op  |> normColumn (*; derived *) |])
    let b = DenseVector.OfEnumerable( M_all.Column(target) )
    let M = DenseMatrix.OfColumnVectors(  M_all.ColumnEnumerator() |> Seq.filter(fun (i,v) -> contains basis i) |> Seq.map snd |> Seq.toArray )

    let x = M.Svd(true).Solve(b)

    let tryPredict (measures:float array) = 
        let a = measures |> Seq.mapi (fun i v -> i,v) |> Seq.filter (fun (i,v) -> contains basis i ) |> Seq.map snd |> Seq.toArray
        let measured = measures |> Seq.mapi (fun i v -> i,v) |> Seq.filter (fun (i,v) -> i = target ) |> Seq.map snd |> Seq.head
        let prediction = DenseMatrix.OfRows(1, a.Length, [| a |]).Multiply(x) |> Seq.head
        prediction, measured, (prediction - measured)/measured

    let p1,m1,e1 = tryPredict (discreteGPUTime |> normColumn)
    let p2,m2,e2 = tryPredict (integratedGPUTime  |> normColumn)
    let p3,m3,e3 = tryPredict (CPUTime |> normColumn)

    // returns errors
    // e1,e2,e3
    // guess
    let predictions = [| 0,p3; 1,p1; 2,p2 |]
    let measures = [| 0,m3; 1,m1; 2,m2 |]
    let bestPredicted,_ = predictions |> Array.sortBy (fun (i,v) -> v) |> Seq.head
    let bestMeasured,_ = measures |> Array.sortBy (fun (i,v) -> v) |> Seq.head
    bestPredicted = bestMeasured
    //(p1-m1)/m1,(p2-m2)/m2,(p3-m3)/m3
    //p1,m1,p2,m2,p3,m3
(*
// raw data
let doCase target size = Array.init 1000 (fun _ -> attempt target size )
let res = doCase 1.0 10
let dev1 = res |> Array.map (fun (a,b,c,d,e,f) -> a,b)
let dev2 = res |> Array.map (fun (a,b,c,d,e,f) -> c,d)
let dev3 = res |> Array.map (fun (a,b,c,d,e,f) -> e,f)

#I @"X:\Projects\EnergonWin8\packages\FSharp.Charting.0.90.6"
#load "FSharp.Charting.fsx"


open FSharp.Charting

Chart.Point dev1

*)

(*
// fun with error percentiles 
//let r1,r2,r3 = Array.init 1000 (fun i -> attempt 3.0 20) |> Array.unzip3

let percentile (p:float) (data:float[]) =
    let sorted = data |> Array.map abs |> Array.sort
    let idx = int(float(sorted.Length) * p)
    sorted.[idx]

let percentiles data =
    percentile 0.25 data, percentile 0.5 data, percentile 0.75 data

let doCase alg size = 
    Array.init 1000 (fun i -> attempt alg size) |> Array.map (fun (a,b,c) -> [|a;b;c|]) |> Array.concat
let r = doCase 5.0 20

percentiles r
*)

// evaluate errors on best device
let doManyAttempts target basissize =
    let res = Array.init 1000 (fun _ -> attempt target basissize) 
    let ok = res |> Array.filter (fun b -> b) |> Array.length
    float(ok)/float(res.Length)

//doManyAttempts 1.0 10

let test () =
    for i in 1..3..40 do
        let ok = doManyAttempts 5.0 i
        System.Console.WriteLine(@"{0} {1}", i, ok)

test()

let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (1 + 2*measures)) )

let sb = new System.Text.StringBuilder()
sb.AppendLine(@"#target basis_size perc_ok")
results |> Array.iteri (fun target row ->row |> Array.iteri (fun measures v -> sb.AppendLine(System.String.Format(@"{0} {1} {2}", target, measures + 1, v)) |> ignore ))
System.IO.File.WriteAllText(System.String.Format(@"{0}\prediction_precision.dat", datafolder), sb.ToString())


// completion time prediction


let attempt target basisSize =

    let basis = Array.init (arithmetic.Length) (fun i -> i, if i = target then 2.0 else rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.take basisSize |> Seq.toArray
    let contains (ary: int array) (value: int) =
        ary |> Array.map (fun v -> v = value) |> Array.exists (id)
    let M_all = DenseMatrix.OfRows(6,arithmetic.Length, [| matrixSize; filterSize; dataTrasfFromHost; dataTrasfFromDevice; readAccessGlobal; arithmetic; |])
    let b = DenseVector.OfEnumerable( M_all.Column(target) )
    let M = DenseMatrix.OfColumnVectors(  M_all.ColumnEnumerator() |> Seq.filter(fun (i,v) -> contains basis i) |> Seq.map snd |> Seq.toArray )

    let x = M.Svd(true).Solve(b)

    let tryPredict (measures:float array) = 
        let a = measures |> Seq.mapi (fun i v -> i,v) |> Seq.filter (fun (i,v) -> contains basis i ) |> Seq.map snd |> Seq.toArray
        let measured = measures |> Seq.mapi (fun i v -> i,v) |> Seq.filter (fun (i,v) -> i = target ) |> Seq.map snd |> Seq.head
        let prediction = DenseMatrix.OfRows(1, a.Length, [| a |]).Multiply(x) |> Seq.head
        prediction, measured, (prediction - measured)/measured

    let p1,m1,e1 = tryPredict discreteGPUTime
    let p2,m2,e2 = tryPredict integratedGPUTime
    let p3,m3,e3 = tryPredict CPUTime

    // returns errors
    e1,e2,e3


// evaluate errors on completion times
let doManyAttempts target basissize =
    let res = Array.init 100 (fun _ -> attempt target basissize) |> Array.map (fun (v1,v2,v3) -> [| v1; v2; v3 |]) |> Array.concat
    let avg = res |> Array.average
    let stddev = res |> Array.map (fun v -> (v-avg)*(v-avg)) |> Array.average
    avg, stddev

doManyAttempts 10 20

let results = Array.init 40 (fun target -> Array.init 39 (fun measures -> doManyAttempts (target) (1 + measures)) )

let sb = new System.Text.StringBuilder()
sb.AppendLine(@"#target basis_size err stddev")
results |> Array.iteri (fun target row ->row |> Array.iteri (fun measures (avg,stddev) -> sb.AppendLine(System.String.Format(@"{0} {1} {2} {3}", target, measures + 1, avg, stddev)) |> ignore ))
System.IO.File.WriteAllText(System.String.Format(@"{0}\prediction_precision2.dat", datafolder), sb.ToString())







let modelsize = 10

let attempt modelsize =

    let orderedCases = Array.init (arithmetic.Length) (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd 
    let basis = orderedCases |> Array.map fst |> Seq.take modelsize |> Seq.toArray
    let targetCase = orderedCases |> Array.rev |> Array.map fst |> Seq.take 1 |> Seq.head
    let contains (ary: int array) (value: int) =
        ary |> Array.map (fun v -> v = value) |> Array.exists (id)
    let M_all = DenseMatrix.OfColumns(arithmetic.Length, 6, [| matrixSize; filterSize; dataTrasfFromHost; dataTrasfFromDevice; readAccessGlobal; arithmetic; |])
    let M = DenseMatrix.OfRowVectors(  M_all.RowEnumerator() |> Seq.filter(fun (i,v) -> contains basis i) |> Seq.map snd |> Seq.toArray )

    //let targetData = discreteGPUTime
    //let targetData = integratedGPUTime
    //let targetData = CPUTime
    let tryPredict (targetData:float[]) = 
        let b = DenseVector.OfEnumerable( targetData |> Seq.mapi (fun i v -> i,v) |> Seq.filter(fun (i,v) -> contains basis i) |> Seq.map snd |> Seq.toArray )
        //b.Count
        let x = M.Svd(true).Solve(b)

        let a = M_all.Row(targetCase)
        let pred = DenseMatrix.OfRows(1, a.Count, [| a |]).Multiply(x) |> Seq.head
        let meas = targetData.[targetCase]
        pred, meas


    let p1,m1 = tryPredict discreteGPUTime
    let p2,m2 =  tryPredict integratedGPUTime
    let p3,m3 =  tryPredict CPUTime

    let predictions = [| p1; p2; p3 |]
    let measures = [| m1; m2; m3 |]

    let bestPrediction = predictions |> Array.mapi (fun i v -> i,v) |> Array.sortBy snd |> Array.map fst |> Seq.head
    let bestMeasured = measures |> Array.mapi (fun i v -> i,v) |> Array.sortBy snd |> Array.map fst |> Seq.head

    bestPrediction = bestMeasured

let experimentPredictDevice modelsize =
    let tot = 1000
    let attempts = Array.init tot (fun i -> attempt modelsize) |> Array.filter (fun v -> v) |> Array.length
    float(attempts)/float(tot)

experimentPredictDevice 5
experimentPredictDevice 10
experimentPredictDevice 15
experimentPredictDevice 20
experimentPredictDevice 25
experimentPredictDevice 30


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

let pseudoinverse (M:Matrix) =
  let D = M.Svd(true)
  let W = D.W()
  let s = D.S()
  let tolerance = Precision.EpsilonOf(2.0) * float(Math.Max(M.RowCount, M.ColumnCount)) * W.[0, 0]
  for i = 0 to s.Count-1 do
    s.[i] <- if s.[i] <= tolerance then 0.0 else 1.0 / s.[i]
  W.SetDiagonal(s)
  (D.U() * W * D.VT()).Transpose()

let isfinite x =
      not (Double.IsNaN(x) || Double.IsInfinity(x))

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

let contains (ary: int array) (value: int) =
  ary |> Array.map (fun v -> v = value) |> Array.exists (id)

let randomizedTargetBasis targetAlg basisSize =
    // filter out cases with work size too little, as .net overhead is larger than actual completion time
    let lowerWorksize = 1024.0*1024.0
    let filtered = workSize |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> v >= lowerWorksize) |> Array.map fst
    let filteredAlg = alg |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> contains filtered i) |> Seq.map snd |> Seq.toArray
    // split cases into belonging to target algorithm or not
    let casesTargetAlg = filteredAlg |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> v=targetAlg) |> Array.map fst
    // pick a target case belonging to target algorithm
    let target = casesTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.head
    let casesNoTargetAlg = filteredAlg |> Array.mapi (fun i v -> i,v) |> Array.filter (fun (i,v) -> v <> targetAlg || true) |> Array.map fst
    // pick a random base to build the model with
    let basis = casesNoTargetAlg |> Array.map (fun i -> i, rnd.NextDouble()) |> Array.sortBy snd |> Array.map fst |> Seq.take basisSize |> Seq.toArray
    target, basis

let attempt target basis =
    let M_all = DenseMatrix.OfRows(4,arithmetic.Length, [| //dataTrasfFromHost |> normColumn; 
                                                            dataTrasfFromDevice |> normColumn; 
                                                            readAccessGlobal |> normColumn; 
                                                            simple_op |> normColumn; 
                                                            complex_op  |> normColumn (*; derived *) |])
    let b = DenseVector.OfEnumerable( M_all.Column(target) )
    let M = DenseMatrix.OfColumnVectors(  M_all.ColumnEnumerator() |> Seq.filter(fun (i,v) -> contains basis i) |> Seq.map snd |> Seq.toArray )

    let cond = M.ConditionNumber()
    let x = M.Svd(true).Solve(b)
    let inv = pseudoinverse M
    //printfn "%A" basis
    //printfn "%A\n%A\n%A\n%A" (M.ConditionNumber()) (M.ToString()) (x) ((inv*b))

    let tryPredict (measures:float array) = 
        let a = measures |> Seq.mapi (fun i v -> i,v) |> Seq.filter (fun (i,v) -> contains basis i ) |> Seq.map snd |> Seq.toArray
        let measured = measures |> Seq.mapi (fun i v -> i,v) |> Seq.filter (fun (i,v) -> i = target ) |> Seq.map snd |> Seq.head
        let origprediction = DenseMatrix.OfRows(1, a.Length, [| a |]).Multiply(x) |> Seq.head
        let prediction = DenseMatrix.OfRows(1, a.Length, [| a |]).Multiply(inv).Multiply(b) |> Seq.head
        //let prediction = origprediction
        //printfn "%A %A" origprediction prediction
        prediction, measured, (prediction - measured)/measured

    let p1,m1,e1 = tryPredict (discreteGPUTime |> normColumn)
    let p2,m2,e2 = tryPredict (integratedGPUTime  |> normColumn)
    let p3,m3,e3 = tryPredict (CPUTime |> normColumn)

    // returns errors
    // e1,e2,e3
    // guess
    let predictions = [| 0,p1; 1,p2; 2,p3 |]
    let measures = [| 0,m1; 1,m2; 2,m3 |]
    let bestPredicted,_ = predictions |> Array.sortBy (fun (i,v) -> v) |> Seq.head
    let bestMeasured,_ = measures |> Array.sortBy (fun (i,v) -> v) |> Seq.head
    //bestPredicted = bestMeasured
    //(p1-m1)/m1,(p2-m2)/m2,(p3-m3)/m3
    //p1,m1,p2,m2,p3,m3
    //isfinite p1 && isfinite p2 && isfinite p3
    //cond
    { M = M; p1 = p1; p2 = p2; p3 = p3; m1 = m1; m2 = m2; m3 = m3; cond = cond; }

let doManyAttempts target basissize =
    let res = Array.init 100 (fun _ -> attempt target basissize)
    let resm1 = Array.map 
    let avg = res |> Array.average
    let prod = res |> Array.map abs |> Array.reduce (*)
    let geomAverage = Math.Pow(prod, 1.0 / float(res.Length))
    let stddev = res |> Array.map (fun v -> (v-avg)*(v-avg)) |> Array.average |> sqrt
    (avg, stddev, res.Length) //, stddev
    //geomAverage
let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (1 + 2*measures)) )


// BOOL
let doManyAttempts target basissize =
    let res = Array.init 100 (fun _ -> attempt target basissize) 
    let ok = res |> Array.filter (fun b -> b) |> Array.length
    float(ok)/float(res.Length)
let results = Array.init 5 (fun target -> Array.init 10 (fun measures -> doManyAttempts (float(target+1)) (1 + 2*measures)) )

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

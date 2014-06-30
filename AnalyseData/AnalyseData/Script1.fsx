open System

let datafolder = @"X:\Projects\featureBasedScheduling\Dati"

// merge convolution files
let csvfiles = System.IO.Directory.EnumerateFiles(datafolder, "Convolution*.csv")
//let handleFile filename =
let filename = csvfiles |> Seq.head
let fieldNames =
    let lines = System.IO.File.ReadAllLines(filename)
    let header = lines |> Seq.head
    header.Split([| ';' |], StringSplitOptions.RemoveEmptyEntries)

let handleFile filename =
    let lines = System.IO.File.ReadAllLines(filename)
    let header = lines |> Seq.head
    let fields = header.Split([| ';' |], StringSplitOptions.RemoveEmptyEntries)
    let handleLine (line:string) =
        let cells = line.Split([| ';' |], StringSplitOptions.RemoveEmptyEntries)
        cells |> Array.map (fun f -> System.Double.Parse(f))
    lines |> Seq.skip 1 |> Seq.map handleLine

let allfields = csvfiles |> Seq.map handleFile |> Seq.concat |> Seq.toArray

fieldNames |> Array.iteri (fun i v -> System.Console.WriteLine(@"{0} {1}", i, v))

let getColumn i = allfields |> Array.map (fun ary -> ary.[i])

let discreteGPUTime = getColumn 0
let integratedGPUTime = getColumn 1
let CPUTime = getColumn 2

let matrixSize = getColumn 3
let filterSize = getColumn 4

let workSize = allfields |> Array.map (fun ary -> ary.[5] * ary.[6] * ary.[7])

let dataTrasfFromHost = getColumn 11
let dataTrasfFromDevice = getColumn 12

let readAccessGlobal = getColumn 15
let writeAccessGlobal = getColumn 16
let readAccessLocal = getColumn 17
let writeAccessLocal = getColumn 18
let readAccessConstant = getColumn 19

let arithmetic = getColumn 20

#r @"X:\Projects\featureBasedScheduling\AnalyseData\packages\MathNet.Numerics.FSharp.2.6.0\lib\net40\MathNet.Numerics.FSharp.dll"
#I @"X:\Projects\featureBasedScheduling\AnalyseData\packages\MathNet.Numerics.2.6.2\lib\net40"
#r @"MathNet.Numerics.dll"
#r @"MathNet.Numerics.IO.dll"
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra.Double

let target = 2

let basisSize = 2

let rnd = new System.Random()

// best device prediction

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
    // e1,e2,e3
    // guess
    let predictions = [| 0,p1; 1,p2; 2,p3 |]
    let measures = [| 0,m1; 1,m2; 2,m3 |]
    let bestPredicted,_ = predictions |> Array.sortBy (fun (i,v) -> v) |> Seq.head
    let bestMeasured,_ = measures |> Array.sortBy (fun (i,v) -> v) |> Seq.head
    bestPredicted = bestMeasured

// evaluate errors on best device
let doManyAttempts target basissize =
    let res = Array.init 1000 (fun _ -> attempt target basissize) 
    let ok = res |> Array.filter (fun b -> b) |> Array.length
    float(ok)/float(res.Length)

let test n =
    for i in 1..40 do
        let ok = doManyAttempts 10 n
        System.Console.WriteLine(@"{0} {1}", i, ok)

test 39

let results = Array.init 40 (fun target -> Array.init 39 (fun measures -> doManyAttempts (target) (1 + measures)) )

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







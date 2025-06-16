const bot = BotManager.getCurrentBot();

let getEnv = require("getEnv.js");
var { KakaoApiService, KakaoShareClient } = require('kakaolink');

const service = KakaoApiService.createService();
const client = KakaoShareClient.createClient();

const cookies = service.login({
    signInWithKakaoTalk: true,
    context: App.getContext(),
}).awaitResult();

client.init(getEnv("mcs_key"), 'https://mcs.shilvister.net', cookies);

isFetchingMsg = undefined

function fetchMsg() {
    try {
        response = org.jsoup.Jsoup.connect('http://172.30.1.28:3003/mcs/alert/fetch')
            .ignoreContentType(true)
            .ignoreHttpErrors(true)
            .timeout(200000)
            .get().text()
        parsedResponse = JSON.parse(response)
        
        for (let i = 0; i < parsedResponse.users.length; i++) {
            const user = parsedResponse.users[i];
            client.sendLink(user, {
                templateId: 120252,
                templateArgs: {"USERID": user.id, "USERNAME": user.name},
            }, 'custom');
        }
    } catch (error) {}
    return
}

function startInterval() {
    isFetchingMsg = setInterval(fetchMsg, 2000)
    return "인터벌 동작이 시작되었습니다."
}

function stopInterval() {
    if (isFetchingMsg !== undefined) {
        clearInterval(isFetchingMsg)
        isFetchingMsg = undefined
    }
    return "인터벌 동작이 중지되었습니다."
}

function onCompile() {
    stopInterval()
}
startInterval()

/*
    (string) msg.content: 메시지의 내용
    (string) msg.room: 메시지를 받은 방 이름
    (User) msg.author: 메시지 전송자
    (string) msg.author.name: 메시지 전송자 이름
    (Image) msg.author.avatar: 메시지 전송자 프로필 사진
    (string) msg.author.avatar.getBase64()
    (boolean) msg.isDebugRoom: 디버그룸에서 받은 메시지일 시 true
    (boolean) msg.isGroupChat: 단체/오픈채팅 여부
    (string) msg.packageName: 메시지를 받은 메신저의 패키지명
    (void) msg.reply(string): 답장하기
    (string) msg.command: 명령어 이름
    (Array) msg.args: 명령어 인자 배열
*/

function onMessage(msg) {
    room = msg.room
    sender = msg.author.name
    content = msg.content

    if (sender != "서건우") return

    // Exception
    if (content.includes("//")) return

    // manageBot
    if (content == "리로드" || content.includes("방해금지")) return

    // Interval
    if (sender == "서건우" && content == "mcs 시작") {
        msg.reply(startInterval())
        return
    }
    
    if (sender == "서건우" && content == "mcs 중지") {
        msg.reply(stopInterval())
        return
    }
}

function onCommand(msg) {
    room = msg.room
    sender = msg.author.name
    content = msg.content
    command = msg.command
    args = msg.args
}

function onCreate(savedInstanceState, activity) {}

function onStart(activity) {}

function onResume(activity) {}

function onPause(activity) {}

function onStop(activity) {}

function onRestart(activity) {}

function onDestroy(activity) {}

function onBackPressed(activity) {}

bot.setCommandPrefix("@")

bot.addListener(Event.MESSAGE, onMessage)
bot.addListener(Event.COMMAND, onCommand)
bot.addListener(Event.START_COMPILE, onCompile)

bot.addListener(Event.Activity.CREATE, onCreate)
bot.addListener(Event.Activity.START, onStart)
bot.addListener(Event.Activity.RESUME, onResume)
bot.addListener(Event.Activity.PAUSE, onPause)
bot.addListener(Event.Activity.STOP, onStop)
bot.addListener(Event.Activity.RESTART, onRestart)
bot.addListener(Event.Activity.DESTROY, onDestroy)
bot.addListener(Event.Activity.BACK_PRESSED, onBackPressed)
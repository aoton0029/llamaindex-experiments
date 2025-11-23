import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
# from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.prompts import PromptTemplate
from factories import (
    LLMFactory,
    ExtractorFactory
)
from services.config_manager import ConfigManager
from factories.template_prompts import TemplatePromptSettings
from llama_index.core import Settings


def test_template(llm: BaseLLM, template: PromptTemplate, template_name: str, **kwargs):
    """
    指定されたテンプレートをLLMに送信してテストする
    """
    print(f"\n{'='*80}")
    print(f"テスト: {template_name}")
    print(f"{'='*80}")
    
    try:
        # テンプレートをフォーマット
        prompt = template.format(**kwargs)
        print("-"*80)
        print(f"【送信プロンプト】\n{prompt[:300]}")
        print("-"*80)
        
        response = llm.complete(prompt)
        
        print("-"*80)
        print(f"【LLMレスポンス】")
        print(f"{response.text}")
        print("-"*80)
        print(f"✓ テスト成功")
        
        return True
        
    except Exception as e:
        print(f"\n✗ テスト失敗: {str(e)}")
        return False

def test_title(llm, context_str: str):
    return test_template(
        llm,
        TemplatePromptSettings.JP_TITLE_NODE_TEMPLATE,
        "Title",
        context_str=context_str
    )
    
def test_summary(llm, context_str: str):
    return test_template(
        llm,
        TemplatePromptSettings.JP_SUMMARY_EXTRACT_TEMPLATE,
        "Summary",
        context_str=context_str
    )

def test_title_extracter(llm, context_str: str):
    extractor = ExtractorFactory.create_extractor(
        "structured_title",
    )
    print(f"\n{'='*80}")
    print("テスト: Title Extractor")
    print(f"{'='*80}")
    try:
        result = extractor.extract(context_str)
        print("-"*80)
        print(f"【Extracted Titles】")
        for title in result:
            print(f"- {title}")
        print("-"*80)
        print(f"✓ テスト成功")
        return True
    except Exception as e:
        print(f"\n✗ テスト失敗: {str(e)}")
        return False


def main():
    """メイン関数"""
    config_manager = ConfigManager("/workspace/src/config")
    TemplatePromptSettings.initialize(config_manager)
    # LLM設定（環境に応じて変更してください）
    backend = "vllm" 
    model_name = "/models/Qwen/Qwen3-8B-AWQ"
    base_url = "http://vllm-llm:8000/v1"
    
    print("LLMの初期化中...")
    try:
        llm = LLMFactory.create(
            backend=backend,
            model_name=model_name,
            base_url=base_url,
            temperature=0.0,
            request_timeout=120.0
        )
        Settings.llm = llm
        print("✓ LLM初期化完了")
    except Exception as e:
        print(f"✗ LLM初期化失敗: {str(e)}")
        return
    
    # テストデータ
    context_str = """
NWH(ノーウェイホーム)が公開されスパイダーマンが世界的に大人気だ。スパイダーマンは元々人気のあるキャラクターだが、MCU版の人気は特にすさまじい。こんなにも愛されるスパイダーマン、そしてMCU版の魅力の正体とは一体なんなのだろうか。

スパイダーマンの魅力で最初に浮かぶのは、月並みだがやはり"共感できる親愛なる隣人"だろう。(いわゆるオリジンにおける)ピーター・パーカーのリアルな身勝手さはそのまま等身大の人間らしさとなり、彼を高潔なヒーローではなく親愛なる隣人として読者視聴者の身近に感じさせる。

ちなみに身勝手さというのは強盗を逃してベンおじさんを死なせたことだけでは無い。大いなる力と大いなる責任というヒーロー性を手に入れてからも、ピーター・パーカーは基本的に他人の意見にあまり耳を貸さず一人で行動するタイプだ。その点でMCU版(ジョン・ワッツ版)のスパイダーマンは人の意見を気にしすぎるとても珍しいスパイダーマンだったといえる。

2017年に新しく生まれ変わったスパイダーマン(MCU版)は、泥棒を捕まえ、迷子のお婆さんを助け、まさしく親愛なる隣人として振る舞った。ただ一点「なんの為の善行か」というスパイダーマンらしからぬあやふやさを抱えたまま。
ピーターに善意が無かったと言いたいわけではない。重要なのは彼が常にトニー・スタークという他人の評価を意識していたことだ。

道に迷ったお婆さんを助けてお礼をもらう、という本来ならちょっと嬉しくなるような出来事も、トニー(ハッピー)の評価がつかなければ喜ぶ気持ちにはなれない。ヴァルチャーを追ってタンカーまで行ったのは事件を未然に防ぐ為だが、ピーターはその動機を「ヴァルチャーをつかまえたら僕も…！(新生アベンジャーズに入れてもらえる！)」と興奮気味に語っている。MCUピーターは人一倍の善人だが、その善行には常に高評価を求める気持ちがついてくる。自分の行動の価値を他人の評価や報酬によって測っているのだ。

HC(ホームカミング)公開当時、英語版の公式アカウントが「何者かになることを夢みる少年」というキャッチコピーをツイートしていて、HCにピッタリのフレーズだなと思った記憶がある。だが、何者かになりたいというのもまた数多のコミックや過去の映画、既存のスパイダーマン達にはほぼ見られない特徴だ。なぜならスパイダーマンとは、力を正しく使わずベンおじさんを亡くしたピーターが、激しい後悔から何者かになった(なることを決意した)結果生まれたヒーローなのだから。本来何者かになりたいではなく、何者かにならざるを得なかったのがスパイダーマン/ピーター・パーカーなのだ。

ところで「評価を気にせずにいられない」「何者かになりたい」というのはSNS時代の多くの若者が抱えるジレンマではないだろうか。その視点で考えると、MCU版スパイダーマンの特異性の中身が見えてくる。

つまりホームカミングにおけるピーターはSNSで自己表現をさぐる若者の特徴を内包しており、注目を集めることに前のめりなスパイダーマンと消極的なピーターという不思議なちぐはぐさも、フォロワー急増加アカウントのスパイダーマンとそれを運営するピーターという二面性として捉えることができる。

HCピーターにとって、スパイダーマンはあくまでインフルエンサーとしてのアバターなのだ。だからこそピーターはリズに正体を明かすことを恐れている。憧れの人気アカウントの正体がどこにでもいるふつうの男の子だと知った時、リズはがっかりしないだろうか？
学校でのピーターはフラッシュにいじめられるパッとしない少年だ(少なくともピーター自身は学校生活をパッとしないものだと感じている)。突然数十万のフォロワーを抱えた15歳の少年が、フォロワーの1人であり淡い恋心を抱く先輩にカミングアウトして、個人アカウントだけフォローされなかったときの気持ちを想像してほしい。アバターのスパイダーマンが愛されていればいるほど、本当の自分を否定されるショックは大きくなる。

スパイダーマンとピーターの関係を仮想現実に例えるのは突飛に聞こえるかもしれないが、実際MCU版スパイダーマンにはシリーズ通して一貫したSNSアカウントの振る舞いを見ることができる。

シビルウォー
地元活動用に作ったスパイダーマンのアカウントが突然セレブにフォロー＆仕事依頼され、ピーターは大いに浮かれながら依頼された役割をこなす
ホームカミング
CWの興奮が冷めやらずセレブの仲間入りを果たすため奮闘するが、急増化したフォロワーの目とイイねを気にしすぎて本来の活動指針を見失う。大きな失敗を糧に自分と向き合ったピーターは、評価に依存せずに選択することの重要さと責任を知る
ファーフロムホーム
インフルエンサーとしての自覚を持ち、責任ある行動ができるようになったスパイダーマン/ピーター。知名度ゆえにプロモーションに利用されそうになるが、それが悪質な自演である事に気付く。主犯を止めることに成功するも、逆にスパイダーマンこそ自演の主の極悪人とするデマによって炎上する
ノーウェイホーム
デマがフェイクニュースとなり炎上が加速。アンチによって個人情報が晒される。拡散された個人情報を消そうと非合法な手段を使うが失敗し、それが呼び水となって更に過激なアンチ達を呼び寄せてしまう。スパイダーマン/ピーターは事態の沈静化とひきかえにアカウントの完全削除を決意する

ほかにも、インフルエンサーのアンチはほとんどの場合本人と面識のない他人(MCUピーターの知らないヴィラン達)、アンチの攻撃はセレブ/人種/性別などの属性にあつまる(スパイダーマンという属性を攻撃したヴィラン達)、悪意あるデマをひろめているのは声が大きいごく一部のアンチ(JJJ)、というSNSの炎上との符合も見ることができる。

スパイダーマンをSNSアカウントとする構図が意図的に仕組まれたものなのか偶然似通ったものなのかを知る術はないが、ジョン・ワッツ率いる制作チームが新シリーズの構築にあたって重要視したのが"スパイダーマンの人気の理由"だったのは間違いないのではないか。

冒頭でも書いたとおり、スパイダーマンの魅力の根幹は「共感を呼ぶ親愛なる隣人」だ。オリジンのピーター・パーカーはもちろん魅力的なキャラクターだが、60年経った今もかつての等身大がそのまま共感を呼ぶキャラクターになるかどうかはわからない。
大衆がもっとも共感できる性質を見つめなおした結果、現代人と切っても切り離せない「SNS」の感覚を持ったまったく新しいスパイダーマンが生まれたのだとしたらそれはとても興味深いし、同時に誕生から約60年を迎えるキャラクターが多くの人の手で更新され、輝き続けていることに感動する。

NWHをもってスパイダーマン／ピーター・パーカーはSNS的側面を持つレガシーをいったん終幕させた。もしも今後も続編が作られるなら、次はどんな「共感できるヒーロー像」が見られるのだろう。シリーズがひと段落したばかりで言うことではないかもしれないが、次回作がとても楽しみだ。
    """
    
    test_results = []    
    test_results.append(test_title(llm, context_str))
    test_results.append(test_summary(llm, context_str))
    test_results.append(test_title_extracter(llm, context_str)) 
    # 結果サマリー
    print(f"\n{'='*80}")
    print("テスト結果サマリー")
    print(f"{'='*80}")
    print(f"成功: {sum(test_results)}/{len(test_results)}")
    print(f"失敗: {len(test_results) - sum(test_results)}/{len(test_results)}")


if __name__ == "__main__":
    main()
